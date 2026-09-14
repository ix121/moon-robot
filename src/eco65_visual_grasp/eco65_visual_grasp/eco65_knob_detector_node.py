import collections
import json
import math
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Movej, Movejp, Movel, Sixforce
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from eco65_visual_grasp.eco65_visual_grasp_node import (
    as_bool,
    matrix_from_transform,
    matrix_from_xyz_rpy,
    pose_from_matrix,
)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - reported at runtime.
    YOLO = None

try:
    import serial
except ImportError:  # pragma: no cover - reported at runtime.
    serial = None

try:
    from orbbec_camera_msgs.msg import Extrinsics
except ImportError:  # pragma: no cover
    Extrinsics = None


KEYPOINT_NAMES = ('corner_1', 'corner_2', 'corner_3', 'corner_4', 'center')
KEYPOINT_COLORS = (
    (255, 80, 80),
    (80, 180, 255),
    (80, 255, 80),
    (255, 80, 255),
    (0, 255, 255),
)


def normalize(vector, fallback=None):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        if fallback is None:
            return None
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def load_handeye(path):
    with open(path, 'r') as stream:
        data = json.load(stream)
    if 'H' in data:
        matrix = np.asarray(data['H'], dtype=float)
    else:
        transform = data['transform']
        t = transform['translation_m']
        q = transform['quaternion_xyzw']
        from eco65_visual_grasp.eco65_visual_grasp_node import matrix_from_quaternion_translation
        matrix = matrix_from_quaternion_translation(q, [t['x'], t['y'], t['z']])
    if matrix.shape != (4, 4):
        raise RuntimeError('handeye matrix must be 4x4')
    return matrix


def ray_from_pixel(camera_info, uv):
    k = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
    return normalize(np.array([
        (float(uv[0]) - k[0, 2]) / k[0, 0],
        (float(uv[1]) - k[1, 2]) / k[1, 1],
        1.0,
    ]))


def ray_plane_intersection(ray, point_on_plane, normal):
    denom = float(np.dot(normal, ray))
    if abs(denom) < 1e-7:
        return None
    scale = float(np.dot(normal, point_on_plane) / denom)
    if scale <= 0.0:
        return None
    return ray * scale


def midpoint(first, second):
    return (np.asarray(first, dtype=float) + np.asarray(second, dtype=float)) * 0.5


def diagonal_intersection(corners):
    first, second, third, fourth = [
        np.asarray(point, dtype=float) for point in corners
    ]
    line_a = third - first
    line_b = second - fourth
    matrix = np.column_stack((line_a, -line_b))
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-9:
        return np.mean(corners, axis=0)
    t, _ = np.linalg.solve(matrix, fourth - first)
    return first + t * line_a


def signed_angle_180(axis):
    return math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180.0


def axis_spread_deg(axes, directed=True):
    """Return the largest angular deviation from the mean axis."""
    vectors = [normalize(axis) for axis in axes]
    vectors = [axis for axis in vectors if axis is not None]
    if not vectors:
        return float('inf')

    reference = vectors[0]
    aligned = []
    for axis in vectors:
        if not directed and float(np.dot(axis, reference)) < 0.0:
            axis = -axis
        aligned.append(axis)

    mean_axis = normalize(np.sum(aligned, axis=0))
    if mean_axis is None:
        return float('inf')
    return max(
        math.degrees(math.acos(np.clip(
            float(np.dot(axis, mean_axis)), -1.0, 1.0)))
        for axis in aligned)


def fit_plane_ransac(points, iterations, distance_threshold):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None

    generator = np.random.default_rng(0)
    best_inliers = None
    best_error = float('inf')
    for _ in range(max(1, int(iterations))):
        sample = points[generator.choice(len(points), 3, replace=False)]
        normal = normalize(np.cross(sample[1] - sample[0], sample[2] - sample[0]))
        if normal is None:
            continue
        distances = np.abs((points - sample[0]).dot(normal))
        inliers = distances <= distance_threshold
        count = int(np.count_nonzero(inliers))
        if count < 3:
            continue
        error = float(np.mean(distances[inliers]))
        if (best_inliers is None or count > int(np.count_nonzero(best_inliers)) or
                (count == int(np.count_nonzero(best_inliers)) and error < best_error)):
            best_inliers = inliers
            best_error = error

    if best_inliers is None:
        return None
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = normalize(vh[-1])
    if normal is None:
        return None
    residuals = np.abs((inlier_points - centroid).dot(normal))
    return {
        'centroid': centroid,
        'normal': normal,
        'inlier_ratio': float(len(inlier_points)) / float(len(points)),
        'rms': float(np.sqrt(np.mean(np.square(residuals)))),
        'inlier_count': int(len(inlier_points)),
    }


class Eco65KnobDetector(Node):
    def __init__(self):
        super().__init__('eco65_knob_detector')
        self.bridge = CvBridge()
        self.motion_lock = threading.Lock()
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._declare_parameters()
        self.base_frame = self.get_parameter('base_frame').value
        self.flange_frame = self.get_parameter('flange_frame').value
        self.flange_to_camera = load_handeye(self.get_parameter('handeye_file').value)
        self.tool_transform = matrix_from_xyz_rpy(
            self.get_parameter('tool_xyz').value,
            self.get_parameter('tool_rpy').value)

        self.color_msg = None
        self.depth_msg = None
        self.color_info = None
        self.depth_info = None
        self.depth_to_color = None
        self.last_process_time = 0.0
        self.history = collections.deque(maxlen=int(self.get_parameter('stable_frames').value))
        self.stable_target = None
        self.last_pnp_approach_b = None
        self.joint_positions = None
        self.joint_state_time = 0.0
        self.turn_stop_event = threading.Event()
        self.mz_samples = collections.deque(maxlen=500)
        self.mz_sequence = 0
        self.mz_baseline = None
        self.mz_triggered_value = None
        self.mz_guard_ready_time = 0.0

        self.model = None
        self.model_error = ''
        if YOLO is None:
            self.model_error = 'python package ultralytics is missing'
        else:
            try:
                self.model = YOLO(str(self.get_parameter('weights').value), task='pose')
            except Exception as exc:
                self.model_error = f'failed to load YOLO model: {exc}'

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Image, self.get_parameter('color_image_topic').value,
            self.color_cb, qos, callback_group=self.cb_group)
        self.create_subscription(
            Image, self.get_parameter('depth_image_topic').value,
            self.depth_cb, qos, callback_group=self.cb_group)
        self.create_subscription(
            CameraInfo, self.get_parameter('color_info_topic').value,
            self.color_info_cb, qos, callback_group=self.cb_group)
        self.create_subscription(
            CameraInfo, self.get_parameter('depth_info_topic').value,
            self.depth_info_cb, qos, callback_group=self.cb_group)
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self.joint_state_cb, qos, callback_group=self.cb_group)
        self.create_subscription(
            Sixforce, self.get_parameter('sixforce_topic').value,
            self.sixforce_cb, qos, callback_group=self.cb_group)
        if not as_bool(self.get_parameter('depth_registered_to_color').value):
            if Extrinsics is None:
                self.get_logger().warn(
                    'orbbec_camera_msgs/Extrinsics is not importable; '
                    'set depth_registered_to_color:=true if depth is already aligned.')
            else:
                extrinsics_qos = QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL)
                self.create_subscription(
                    Extrinsics,
                    self.get_parameter('depth_to_color_topic').value,
                    self.depth_to_color_cb,
                    extrinsics_qos,
                    callback_group=self.cb_group)

        from geometry_msgs.msg import PoseStamped
        self.pose_pub = self.create_publisher(PoseStamped, '~/target_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.movejp_pub = self.create_publisher(Movejp, self.get_parameter('movej_p_topic').value, 10)
        self.movel_pub = self.create_publisher(Movel, self.get_parameter('movel_topic').value, 10)
        self.movej_pub = self.create_publisher(Movej, self.get_parameter('movej_topic').value, 10)
        self.move_stop_pub = self.create_publisher(Empty, self.get_parameter('move_stop_topic').value, 10)
        self.movej_result = None
        self.movejp_result = None
        self.movel_result = None
        self.cached_grasp_target = None
        self.create_subscription(
            Bool, self.get_parameter('movej_p_result_topic').value,
            self.movejp_result_cb, 10, callback_group=self.cb_group)
        self.create_subscription(
            Bool, self.get_parameter('movel_result_topic').value,
            self.movel_result_cb, 10, callback_group=self.cb_group)
        self.create_subscription(
            Bool, self.get_parameter('movej_result_topic').value,
            self.movej_result_cb, 10, callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/open_gripper', self.open_gripper_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/close_gripper', self.close_gripper_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/move_to_observation', self.move_to_observation_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/move_to_pregrasp', self.move_to_pregrasp_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/complete_from_pregrasp', self.complete_from_pregrasp_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/grasp_knob', self.grasp_knob_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/turn_knob', self.turn_knob_cb,
            callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/stop_turning', self.stop_turning_cb,
            callback_group=self.cb_group)
        self.create_timer(0.05, self.process, callback_group=self.cb_group)

        self.get_logger().info(
            'ECO65 knob detector ready. model=%s, base=%s, flange=%s' % (
                self.get_parameter('weights').value, self.base_frame, self.flange_frame))

    def _declare_parameters(self):
        defaults = {
            'base_frame': 'baselink',
            'flange_frame': 'Link6',
            'handeye_file': '/home/proton/eco65_grasp_ws/handeye_config.json',
            'tool_xyz': [0.0, 0.0, 0.155],
            'tool_rpy': [0.0, 0.0, 0.0],
            'weights': '/home/proton/knob/best.onnx',
            'device': 'cpu',
            'imgsz': 960,
            'confidence': 0.50,
            'keypoint_confidence': 0.35,
            'process_hz': 5.0,
            'color_image_topic': '/camera/color/image_raw',
            'color_info_topic': '/camera/color/camera_info',
            'depth_image_topic': '/camera/depth/image_raw',
            'depth_info_topic': '/camera/depth/camera_info',
            'depth_to_color_topic': '/camera/depth_to_color',
            'depth_registered_to_color': False,
            'depth_min_m': 0.08,
            'depth_max_m': 1.20,
            'depth_stride': 2,
            'polygon_shrink': 0.78,
            'min_plane_points': 50,
            'plane_source': 'pnp',
            'pnp_long_size_m': 0.030,
            'pnp_short_size_m': 0.010,
            'pnp_max_reprojection_error_px': 4.0,
            'pnp_expected_approach_axis_base': [0.0, 1.0, 0.0],
            'pnp_max_approach_deviation_deg': 45.0,
            'pnp_continuity_weight_px_per_deg': 0.02,
            'lock_approach_axis_base': True,
            'fixed_approach_axis_base': [0.0, 1.0, 0.0],
            'fixed_plane_depth_source': 'two_surface_midpoint',
            'fixed_plane_depth_offset_m': 0.0,
            'fixed_plane_min_surface_separation_m': 0.015,
            'fixed_plane_max_surface_separation_m': 0.035,
            'fixed_plane_min_cluster_ratio': 0.10,
            'fixed_plane_single_surface_fallback_enabled': True,
            'single_surface_max_separation_m': 0.010,
            'knob_front_to_center_offset_m': 0.005,
            'single_surface_extra_depth_offset_m': -0.015,
            'fixed_plane_panel_fallback_enabled': True,
            'panel_to_knob_center_offset_m': 0.012,
            'panel_fixed_axis_max_angle_deg': 20.0,
            'panel_outer_scale': 2.20,
            'panel_inner_scale': 1.25,
            'panel_min_points': 100,
            'panel_ransac_iterations': 100,
            'panel_inlier_threshold_m': 0.003,
            'panel_min_inlier_ratio': 0.45,
            'panel_max_rms_m': 0.0025,
            'panel_max_view_angle_deg': 70.0,
            'knob_height_percentile': 75.0,
            'knob_min_height_m': 0.002,
            'knob_max_height_m': 0.050,
            'knob_surface_offset_m': 0.0,
            'stable_frames': 5,
            'stable_position_std_m': 0.006,
            'stable_angle_std_deg': 6.0,
            'stable_normal_spread_deg': 5.0,
            'stable_axis_spread_deg': 6.0,
            'target_timeout_s': 5.0,
            'center_error_ratio_max': 0.25,
            'target_center_source': 'model',
            'center_blend_max_difference_m': 0.004,
            'center_reject_difference_m': 0.008,
            'target_position_offset_m': [0.0, 0.0, 0.0],
            'min_long_size_m': 0.020,
            'max_long_size_m': 0.045,
            'min_short_size_m': 0.005,
            'max_short_size_m': 0.018,
            'reject_size_out_of_range': True,
            'pregrasp_distance_m': 0.070,
            'grasp_extra_depth_m': -0.005,
            'retreat_distance_m': 0.040,
            # +1 approaches the visible knob surface from the camera side.
            'approach_axis_sign': 1.0,
            'retreat_after_close': False,
            'grasp_orientation_offset_rpy': [0.0, 0.0, 0.0],
            'use_current_orientation_for_pregrasp': False,
            'execute_motion': False,
            'joint_states_topic': '/joint_states',
            'joint_state_timeout_s': 1.0,
            'observation_joints': [
                1.5079417659759522,
                -0.34245625,
                2.188142691421509,
                -1.2945282073974609,
                -1.5033524074554443,
                -0.03819804810285568,
            ],
            'observation_speed': 8,
            'sixforce_topic': '/rm_driver/udp_six_zero_force',
            'force_sample_timeout_s': 0.20,
            'mz_guard_enabled': True,
            'mz_zero_window_s': 0.50,
            'mz_zero_min_samples': 20,
            'mz_resistance_threshold_nm': 0.40,
            'mz_trigger_samples': 3,
            'mz_guard_delay_s': 0.10,
            'force_release_wait_s': 0.10,
            'movej_topic': '/rm_driver/movej_cmd',
            'movej_result_topic': '/rm_driver/movej_result',
            'movej_p_topic': '/rm_driver/movej_p_cmd',
            'movej_p_result_topic': '/rm_driver/movej_p_result',
            'movel_topic': '/rm_driver/movel_cmd',
            'movel_result_topic': '/rm_driver/movel_result',
            'move_stop_topic': '/rm_driver/move_stop_cmd',
            'pregrasp_use_linear_motion': False,
            'max_pregrasp_distance_m': 0.40,
            'pose_speed': 8,
            'linear_speed': 8,
            'command_timeout_s': 45.0,
            'turn_cycles': 2,
            'turn_until_resistance': True,
            'max_turn_cycles': 10,
            'turn_step_deg': 10.0,
            'turn_steps_per_cycle': 18,
            'reset_step_deg': 90.0,
            'reset_steps': 2,
            'wrist_clockwise_sign': 1.0,
            'wrist_speed': 8,
            'wrist_joint_limit_deg': 350.0,
            'reset_after_last_cycle': False,
            'gripper_port': '/dev/ttyACM0',
            'gripper_baud': 115200,
            'gripper_timeout_s': 1.0,
            'gripper_open_hex': '7b01020020492000c8f97d',
            'gripper_close_hex': '7b01020120492000c8f87d',
            'gripper_motion_wait_s': 1.0,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)

    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text, throttle_duration_sec=2.0)

    def color_cb(self, msg):
        self.color_msg = msg

    def depth_cb(self, msg):
        self.depth_msg = msg

    def color_info_cb(self, msg):
        self.color_info = msg

    def depth_info_cb(self, msg):
        self.depth_info = msg

    def depth_to_color_cb(self, msg):
        rotation = np.asarray(msg.rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(msg.translation, dtype=float).reshape(3)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        self.depth_to_color = matrix

    def lookup_base_to_camera(self):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, self.flange_frame, rclpy.time.Time())
        return matrix_from_transform(transform.transform).dot(self.flange_to_camera)

    def lookup_base_to_flange(self):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, self.flange_frame, rclpy.time.Time())
        return matrix_from_transform(transform.transform)

    def require_motion(self):
        if not as_bool(self.get_parameter('execute_motion').value):
            return 'Motion is disabled; relaunch with execute_motion:=true'
        return None

    def tip_to_flange(self, tip_matrix):
        return tip_matrix.dot(np.linalg.inv(self.tool_transform))

    def current_tip_matrix(self):
        return self.lookup_base_to_flange().dot(self.tool_transform)

    def pregrasp_motion_matrix(self, target):
        tip_matrix = np.array(target['pregrasp_matrix'])
        if not as_bool(self.get_parameter('use_current_orientation_for_pregrasp').value):
            return tip_matrix
        try:
            base_to_flange = self.lookup_base_to_flange()
        except TransformException:
            return tip_matrix
        flange_matrix = self.tip_to_flange(tip_matrix)
        flange_matrix[:3, :3] = base_to_flange[:3, :3]
        return flange_matrix.dot(self.tool_transform)

    def current_stable_target(self):
        target = self.stable_target
        if target is None:
            return None
        age = (self.get_clock().now() - target['stamp']).nanoseconds * 1e-9
        if age > float(self.get_parameter('target_timeout_s').value):
            return None
        return target

    def joint_state_cb(self, msg):
        by_name = {
            str(name).lower(): float(position)
            for name, position in zip(msg.name, msg.position)
        }
        names = [f'joint{index}' for index in range(1, 7)]
        if not all(name in by_name for name in names):
            return
        self.joint_positions = np.asarray([by_name[name] for name in names], dtype=float)
        self.joint_state_time = time.monotonic()

    def sixforce_cb(self, msg):
        now = time.monotonic()
        self.mz_sequence += 1
        self.mz_samples.append((now, float(msg.force_mz), self.mz_sequence))

    def capture_mz_baseline(self):
        window = float(self.get_parameter('mz_zero_window_s').value)
        minimum = int(self.get_parameter('mz_zero_min_samples').value)
        timeout = float(self.get_parameter('force_sample_timeout_s').value)
        deadline = time.monotonic() + max(window, timeout, 0.5)
        samples = []
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            samples = [
                value for stamp, value, _sequence in list(self.mz_samples)
                if now - stamp <= window
            ]
            if (len(samples) >= minimum and self.mz_samples and
                    now - self.mz_samples[-1][0] <= timeout):
                self.mz_baseline = float(np.median(samples))
                self.get_logger().info(
                    f'[力控] Mz baseline={self.mz_baseline:.3f}Nm, '
                    f'samples={len(samples)}')
                return True, 'Mz baseline ready'
            time.sleep(0.01)
        return False, (
            f'Cannot establish fresh Mz baseline: got {len(samples)} samples, '
            f'need {minimum}; check {self.get_parameter("sixforce_topic").value}')

    def current_joints(self):
        if self.joint_positions is None:
            return None, 'No complete /joint_states sample'
        age = time.monotonic() - self.joint_state_time
        timeout = float(self.get_parameter('joint_state_timeout_s').value)
        if age > timeout:
            return None, f'/joint_states is stale: {age:.2f}s > {timeout:.2f}s'
        return np.array(self.joint_positions), ''

    def movej_result_cb(self, msg):
        self.movej_result = bool(msg.data)

    def movejp_result_cb(self, msg):
        self.movejp_result = bool(msg.data)

    def movel_result_cb(self, msg):
        self.movel_result = bool(msg.data)

    def wait_bool_result(self, getter, label):
        deadline = time.monotonic() + float(self.get_parameter('command_timeout_s').value)
        while rclpy.ok() and time.monotonic() < deadline:
            result = getter()
            if result is not None:
                return bool(result), f'{label} result={result}'
            time.sleep(0.05)
        self.move_stop_pub.publish(Empty())
        self.get_logger().warn(f'{label} timed out; sent move_stop_cmd')
        return False, f'{label} result timed out'

    def publish_pose_motion(self, tip_matrix, linear=False):
        flange_matrix = self.tip_to_flange(tip_matrix)
        msg = Movel() if linear else Movejp()
        msg.pose = pose_from_matrix(flange_matrix)
        msg.speed = int(self.get_parameter('linear_speed' if linear else 'pose_speed').value)
        msg.block = True
        msg.trajectory_connect = 0
        p = msg.pose.position
        q = msg.pose.orientation
        self.get_logger().info(
            f'[运动目标] {"MoveL" if linear else "MoveJ_P"} flange xyz='
            f'({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), q='
            f'({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}), '
            f'speed={msg.speed}')
        if linear:
            self.movel_result = None
            self.movel_pub.publish(msg)
            return self.wait_bool_result(lambda: self.movel_result, 'MoveL')
        self.movejp_result = None
        self.movejp_pub.publish(msg)
        return self.wait_bool_result(lambda: self.movejp_result, 'MoveJ_P')

    def wait_movej_result(self, label, monitor_mz):
        deadline = time.monotonic() + float(self.get_parameter('command_timeout_s').value)
        threshold = float(self.get_parameter('mz_resistance_threshold_nm').value)
        required = max(1, int(self.get_parameter('mz_trigger_samples').value))
        timeout = float(self.get_parameter('force_sample_timeout_s').value)
        over_count = 0
        last_sequence = -1

        while rclpy.ok() and time.monotonic() < deadline:
            if self.turn_stop_event.is_set():
                return False, f'{label}: stopped by /stop_turning'
            if monitor_mz and time.monotonic() >= self.mz_guard_ready_time:
                if not self.mz_samples or time.monotonic() - self.mz_samples[-1][0] > timeout:
                    self.move_stop_pub.publish(Empty())
                    return False, f'{label}: sixforce sample became stale'
                stamp, value, sequence = self.mz_samples[-1]
                if sequence != last_sequence:
                    last_sequence = sequence
                    delta = value - float(self.mz_baseline)
                    over_count = over_count + 1 if abs(delta) >= threshold else 0
                    if over_count >= required:
                        self.mz_triggered_value = delta
                        self.move_stop_pub.publish(Empty())
                        self.get_logger().warn(
                            f'[力控触发] Mz delta={delta:.3f}Nm >= '
                            f'{threshold:.3f}Nm，已发送 move_stop')
                        return False, f'{label}: Mz resistance detected ({delta:.3f}Nm)'
            if self.movej_result is not None:
                return bool(self.movej_result), f'{label} result={self.movej_result}'
            time.sleep(0.005)

        self.move_stop_pub.publish(Empty())
        return False, f'{label} result timed out'

    def publish_joint_motion(
            self, joints, label, monitor_mz=False,
            speed_parameter='wrist_speed'):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,):
            return False, f'{label}: expected 6 joints, got {len(joints)}'
        wrist_limit = math.radians(float(self.get_parameter('wrist_joint_limit_deg').value))
        if wrist_limit > 0.0 and abs(float(joints[5])) > wrist_limit:
            return False, (
                f'{label}: joint6 target {math.degrees(joints[5]):.1f}deg exceeds '
                f'+/-{math.degrees(wrist_limit):.1f}deg')
        if self.turn_stop_event.is_set():
            return False, f'{label}: stopped by /stop_turning'

        msg = Movej()
        msg.joint = [float(value) for value in joints]
        msg.speed = int(self.get_parameter(speed_parameter).value)
        msg.block = True
        msg.trajectory_connect = 0
        msg.dof = 6
        self.get_logger().info(
            f'[关节运动] {label}: joint6={math.degrees(joints[5]):.1f}deg, '
            f'speed={msg.speed}')
        self.movej_result = None
        self.movej_pub.publish(msg)
        return self.wait_movej_result(label, monitor_mz)

    def rotate_wrist(self, step_deg, steps, label, monitor_mz=False):
        start_joints, error = self.current_joints()
        if start_joints is None:
            return False, f'{label}: {error}'
        for index in range(int(steps)):
            target = np.array(start_joints)
            target[5] += math.radians(float(step_deg) * float(index + 1))
            ok, text = self.publish_joint_motion(
                target, f'{label} {index + 1}/{int(steps)}',
                monitor_mz=monitor_mz)
            if not ok:
                return False, text
        return True, f'{label} complete'

    def command_gripper(self, closing):
        if serial is None:
            return False, 'python serial module is missing; install python3-serial'
        hex_string = (
            self.get_parameter('gripper_close_hex').value if closing else
            self.get_parameter('gripper_open_hex').value)
        try:
            with serial.Serial(
                self.get_parameter('gripper_port').value,
                int(self.get_parameter('gripper_baud').value),
                timeout=float(self.get_parameter('gripper_timeout_s').value),
            ) as stream:
                stream.write(bytes.fromhex(hex_string))
            time.sleep(float(self.get_parameter('gripper_motion_wait_s').value))
            return True, 'gripper close command sent' if closing else 'gripper open command sent'
        except Exception as exc:
            return False, f'gripper serial command failed: {exc}'

    def open_gripper_cb(self, _request, response):
        response.success, response.message = self.command_gripper(False)
        return response

    def close_gripper_cb(self, _request, response):
        response.success, response.message = self.command_gripper(True)
        return response

    def move_to_observation_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        joints = np.asarray(
            self.get_parameter('observation_joints').value, dtype=float)
        if joints.shape != (6,):
            response.success = False
            response.message = 'observation_joints must contain 6 values'
            return response
        with self.motion_lock:
            self.turn_stop_event.clear()
            ok, text = self.publish_joint_motion(
                joints, 'move to knob observation',
                speed_parameter='observation_speed')
        response.success = ok
        response.message = (
            'moved to knob observation: ' + text if ok else
            'move to knob observation failed: ' + text)
        return response

    def grasp_matrix(self, target):
        matrix = np.array(target['matrix'])
        matrix[:3, 3] = (
            target['position'] +
            target['approach_axis'] *
            float(self.get_parameter('grasp_extra_depth_m').value))
        return matrix

    def retreat_matrix(self, target):
        matrix = np.array(target['matrix'])
        matrix[:3, 3] = (
            target['position'] -
            target['approach_axis'] *
            float(self.get_parameter('retreat_distance_m').value))
        return matrix

    def turn_retreat_position(self, target):
        return (
            self.grasp_matrix(target)[:3, 3] -
            target['approach_axis'] *
            float(self.get_parameter('retreat_distance_m').value))

    def move_to_pregrasp_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            target = self.current_stable_target()
            if target is None:
                response.success = False
                response.message = 'No fresh stable knob target'
                return response
            pregrasp = self.pregrasp_motion_matrix(target)
            try:
                current_tip = self.current_tip_matrix()
                distance = float(np.linalg.norm(pregrasp[:3, 3] - current_tip[:3, 3]))
            except TransformException:
                distance = 0.0
            max_distance = float(self.get_parameter('max_pregrasp_distance_m').value)
            if max_distance > 0.0 and distance > max_distance:
                response.success = False
                response.message = (
                    f'pregrasp target is too far from current TCP: '
                    f'{distance:.3f}m > {max_distance:.3f}m')
                return response
            linear = as_bool(self.get_parameter('pregrasp_use_linear_motion').value)
            self.get_logger().info(
                f'[预抓取] current->pregrasp distance={distance:.3f}m, '
                f'use {"MoveL" if linear else "MoveJ_P"}')
            ok, text = self.publish_pose_motion(pregrasp, linear=linear)
            if ok:
                self.cached_grasp_target = dict(target)
            response.success = ok
            prefix = f'pregrasp distance={distance:.3f}m; '
            response.message = (
                prefix + 'moved to knob pregrasp: ' + text if ok else
                prefix + 'pregrasp failed: ' + text)
            return response

    def complete_from_pregrasp_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            target = self.cached_grasp_target or self.current_stable_target()
            if target is None:
                response.success = False
                response.message = 'No cached knob target from pregrasp'
                return response
            pregrasp_pos = np.asarray(target['pregrasp_matrix'])[:3, 3]
            grasp_pos = self.grasp_matrix(target)[:3, 3]
            retreat_pos = self.retreat_matrix(target)[:3, 3]
            self.get_logger().info(
                '[抓取路径] TCP pregrasp=(%.4f, %.4f, %.4f) -> '
                'grasp=(%.4f, %.4f, %.4f) -> retreat=(%.4f, %.4f, %.4f)' %
                (*pregrasp_pos, *grasp_pos, *retreat_pos))
            ok, text = self.command_gripper(False)
            if not ok:
                response.success = False
                response.message = 'Open failed: ' + text
                return response
            ok, text = self.publish_pose_motion(self.grasp_matrix(target), linear=True)
            if not ok:
                response.success = False
                response.message = 'Linear approach failed: ' + text
                return response
            ok, text = self.command_gripper(True)
            if not ok:
                response.success = False
                response.message = 'Close failed: ' + text
                return response
            if not as_bool(self.get_parameter('retreat_after_close').value):
                response.success = True
                response.message = 'knob grasp complete: gripper closed; retreat skipped'
                return response
            ok, text = self.publish_pose_motion(self.retreat_matrix(target), linear=True)
            response.success = ok
            response.message = 'knob grasp complete: ' + text if ok else 'Retreat failed: ' + text
            return response

    def grasp_knob_cb(self, request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            target = self.current_stable_target()
            if target is None:
                response.success = False
                response.message = 'No fresh stable knob target'
                return response
            pregrasp = self.pregrasp_motion_matrix(target)
            linear = as_bool(self.get_parameter('pregrasp_use_linear_motion').value)
            ok, text = self.publish_pose_motion(pregrasp, linear=linear)
            if not ok:
                response.success = False
                response.message = 'Pregrasp failed: ' + text
                return response
            self.cached_grasp_target = dict(target)
        return self.complete_from_pregrasp_cb(request, response)

    def stop_turning_cb(self, _request, response):
        self.turn_stop_event.set()
        self.move_stop_pub.publish(Empty())
        response.success = True
        response.message = 'turn stop requested; move_stop_cmd sent'
        return response

    def release_and_retreat(self, target, reason):
        self.move_stop_pub.publish(Empty())
        time.sleep(float(self.get_parameter('force_release_wait_s').value))
        ok, text = self.command_gripper(False)
        if not ok:
            return False, reason + '; gripper open failed: ' + text
        try:
            retreat = self.current_tip_matrix()
        except TransformException as exc:
            return False, f'{reason}; gripper opened, but retreat TF failed: {exc}'
        retreat[:3, 3] = self.turn_retreat_position(target)
        ok, text = self.publish_pose_motion(retreat, linear=True)
        if not ok:
            return False, reason + '; gripper opened, but retreat failed: ' + text
        return True, reason + '; gripper opened and retreated'

    def turn_knob_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response

        with self.motion_lock:
            target = self.cached_grasp_target
            if target is None:
                response.success = False
                response.message = 'No cached knob target; complete knob grasp first'
                return response

            until_resistance = as_bool(
                self.get_parameter('turn_until_resistance').value)
            cycles = max(1, int(self.get_parameter(
                'max_turn_cycles' if until_resistance else 'turn_cycles').value))
            turn_step = float(self.get_parameter('turn_step_deg').value)
            turn_steps = max(1, int(self.get_parameter('turn_steps_per_cycle').value))
            reset_step = float(self.get_parameter('reset_step_deg').value)
            reset_steps = max(1, int(self.get_parameter('reset_steps').value))
            clockwise_sign = float(self.get_parameter('wrist_clockwise_sign').value)
            if abs(clockwise_sign) < 0.5:
                response.success = False
                response.message = 'wrist_clockwise_sign must be 1.0 or -1.0'
                return response
            clockwise_sign = 1.0 if clockwise_sign > 0.0 else -1.0
            reset_after_last = as_bool(
                self.get_parameter('reset_after_last_cycle').value)

            self.turn_stop_event.clear()
            self.mz_triggered_value = None
            monitor_mz = as_bool(self.get_parameter('mz_guard_enabled').value)
            if until_resistance and not monitor_mz:
                response.success = False
                response.message = (
                    'turn_until_resistance requires mz_guard_enabled=true')
                return response
            if monitor_mz:
                ok, text = self.capture_mz_baseline()
                if not ok:
                    response.success = False
                    response.message = text
                    return response
                self.mz_guard_ready_time = (
                    time.monotonic() +
                    float(self.get_parameter('mz_guard_delay_s').value))
            self.get_logger().info(
                f'[拧旋钮] 模式={"拧紧力停止" if until_resistance else "固定轮数"}，'
                f'最多 {cycles} 轮；顺时针每轮 '
                f'{turn_steps}x{turn_step:.1f}deg，符号={clockwise_sign:+.0f}')

            for cycle in range(cycles):
                cycle_label = f'cycle {cycle + 1}/{cycles}'
                ok, text = self.rotate_wrist(
                    clockwise_sign * turn_step, turn_steps,
                    f'{cycle_label} clockwise', monitor_mz=monitor_mz)
                if not ok:
                    if self.mz_triggered_value is not None:
                        released, release_text = self.release_and_retreat(
                            target,
                            f'Mz resistance {self.mz_triggered_value:.3f}Nm detected')
                        response.success = released
                        response.message = release_text
                        return response
                    response.success = False
                    response.message = f'{cycle_label} turn failed: {text}'
                    return response

                is_last = cycle == cycles - 1
                if is_last and until_resistance:
                    released, release_text = self.release_and_retreat(
                        target,
                        f'Maximum {cycles} cycles reached without Mz resistance')
                    response.success = False
                    response.message = release_text
                    return response
                if is_last and not reset_after_last:
                    break

                ok, text = self.command_gripper(False)
                if not ok:
                    response.success = False
                    response.message = f'{cycle_label} open failed: {text}'
                    return response

                try:
                    retreat = self.current_tip_matrix()
                except TransformException as exc:
                    response.success = False
                    response.message = f'{cycle_label} retreat TF failed: {exc}'
                    return response
                retreat[:3, 3] = self.turn_retreat_position(target)
                ok, text = self.publish_pose_motion(retreat, linear=True)
                if not ok:
                    response.success = False
                    response.message = f'{cycle_label} retreat failed: {text}'
                    return response

                ok, text = self.rotate_wrist(
                    -clockwise_sign * reset_step, reset_steps,
                    f'{cycle_label} counterclockwise reset')
                if not ok:
                    response.success = False
                    response.message = f'{cycle_label} reset failed: {text}'
                    return response

                ok, text = self.publish_pose_motion(self.grasp_matrix(target), linear=True)
                if not ok:
                    response.success = False
                    response.message = f'{cycle_label} re-approach failed: {text}'
                    return response

                ok, text = self.command_gripper(True)
                if not ok:
                    response.success = False
                    response.message = f'{cycle_label} re-close failed: {text}'
                    return response

            response.success = True
            response.message = (
                f'knob turning complete: {cycles} cycles, '
                f'{turn_steps * turn_step:.1f}deg clockwise per cycle')
            return response

    def process(self):
        now = time.monotonic()
        period = 1.0 / max(float(self.get_parameter('process_hz').value), 0.1)
        if now - self.last_process_time < period:
            return
        self.last_process_time = now

        if self.model is None:
            self.set_status(self.model_error)
            return
        if self.color_msg is None or self.depth_msg is None or self.color_info is None or self.depth_info is None:
            self.set_status('waiting for camera images/info')
            return
        if not as_bool(self.get_parameter('depth_registered_to_color').value) and self.depth_to_color is None:
            self.set_status('waiting for ' + self.get_parameter('depth_to_color_topic').value)
            return

        try:
            color = self.bridge.imgmsg_to_cv2(self.color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='passthrough')
            base_to_camera = self.lookup_base_to_camera()
        except (TransformException, Exception) as exc:
            self.set_status(f'input conversion/TF failed: {exc}')
            return

        detection = self.detect_best(color)
        debug = color.copy()
        if detection is None:
            self.history.clear()
            self.clear_markers()
            self.set_status('no valid knob detection')
            self.publish_debug(debug)
            return

        target, reason = self.localize_target(depth, detection, base_to_camera)
        self.draw_detection(debug, detection, target is not None, reason)
        self.publish_debug(debug)
        if target is None:
            self.history.clear()
            self.clear_markers()
            self.set_status(reason)
            return

        stable = self.update_stability(target)
        self.publish_markers(target, stable is not None)
        if stable is None:
            self.set_status(
                'knob detected; waiting for stability '
                f'({len(self.history)}/{int(self.get_parameter("stable_frames").value)})')
            return

        self.stable_target = stable
        self.publish_pose(stable)
        p = stable['position']
        self.set_status(
            f'stable knob ready in {self.base_frame}: '
            f'[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]')

    def detect_best(self, image):
        result = self.model.predict(
            image,
            conf=float(self.get_parameter('confidence').value),
            imgsz=int(self.get_parameter('imgsz').value),
            device=self.get_parameter('device').value,
            verbose=False,
        )[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is None:
            keypoint_conf = np.ones(keypoints.shape[:2], dtype=np.float32)
        else:
            keypoint_conf = result.keypoints.conf.cpu().numpy()

        for index in np.argsort(-conf):
            xy = np.asarray(keypoints[index], dtype=float)
            kconf = np.asarray(keypoint_conf[index], dtype=float)
            if xy.shape != (5, 2) or kconf.shape != (5,):
                continue
            if float(np.min(kconf)) < float(self.get_parameter('keypoint_confidence').value):
                continue
            geometry_ok, reason = self.check_2d_geometry(xy)
            if not geometry_ok:
                continue
            return {
                'box': np.asarray(boxes[index], dtype=float),
                'box_conf': float(conf[index]),
                'keypoints': xy,
                'keypoint_conf': kconf,
                'geometry_reason': reason,
            }
        return None

    def check_2d_geometry(self, xy):
        h = self.color_info.height
        w = self.color_info.width
        margin = 4.0
        if np.any(xy[:, 0] < margin) or np.any(xy[:, 0] >= w - margin):
            return False, 'keypoint near image edge'
        if np.any(xy[:, 1] < margin) or np.any(xy[:, 1] >= h - margin):
            return False, 'keypoint near image edge'
        corners = xy[:4]
        long_end_1 = midpoint(corners[0], corners[1])
        long_end_2 = midpoint(corners[2], corners[3])
        side_1 = midpoint(corners[1], corners[2])
        side_2 = midpoint(corners[3], corners[0])
        long_px = float(np.linalg.norm(long_end_2 - long_end_1))
        short_px = float(np.linalg.norm(side_2 - side_1))
        if long_px < 8.0 or short_px < 3.0 or long_px <= short_px:
            return False, 'degenerate knob geometry'
        geometric_center = diagonal_intersection(corners)
        center_error = float(np.linalg.norm(xy[4] - geometric_center))
        if center_error > long_px * float(self.get_parameter('center_error_ratio_max').value):
            return False, 'center far from quadrilateral center'
        return True, 'ok'

    def depth_points_in_region(
            self, depth, polygon, outer_scale, inner_scale=None):
        if depth.dtype == np.uint16:
            z_image = depth.astype(np.float32) * 0.001
        else:
            z_image = depth.astype(np.float32)
        stride = max(1, int(self.get_parameter('depth_stride').value))
        rows, cols = np.mgrid[0:z_image.shape[0]:stride, 0:z_image.shape[1]:stride]
        z = z_image[::stride, ::stride].reshape(-1)
        u = cols.reshape(-1).astype(np.float32)
        v = rows.reshape(-1).astype(np.float32)
        valid = np.isfinite(z)
        valid &= z >= float(self.get_parameter('depth_min_m').value)
        valid &= z <= float(self.get_parameter('depth_max_m').value)
        u, v, z = u[valid], v[valid], z[valid]
        if z.size == 0:
            return None

        if as_bool(self.get_parameter('depth_registered_to_color').value):
            k = np.asarray(self.color_info.k).reshape(3, 3)
            points_c = np.vstack(((u - k[0, 2]) / k[0, 0] * z,
                                  (v - k[1, 2]) / k[1, 1] * z, z))
            uc = u.astype(int)
            vc = v.astype(int)
        else:
            kd = np.asarray(self.depth_info.k).reshape(3, 3)
            points_d = np.vstack(((u - kd[0, 2]) / kd[0, 0] * z,
                                  (v - kd[1, 2]) / kd[1, 1] * z, z))
            points_c = self.depth_to_color[:3, :3].dot(points_d) + self.depth_to_color[:3, 3:4]
            positive = points_c[2] > 1e-6
            points_c = points_c[:, positive]
            kc = np.asarray(self.color_info.k).reshape(3, 3)
            uc = np.rint(kc[0, 0] * points_c[0] / points_c[2] + kc[0, 2]).astype(int)
            vc = np.rint(kc[1, 1] * points_c[1] / points_c[2] + kc[1, 2]).astype(int)

        inside = ((uc >= 0) & (vc >= 0) & (uc < self.color_info.width) & (vc < self.color_info.height))
        uc, vc = uc[inside], vc[inside]
        points_c = points_c[:, inside]

        polygon = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
        center = polygon.mean(axis=0)
        outer = center + float(outer_scale) * (polygon - center)
        mask = np.zeros((self.color_info.height, self.color_info.width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(outer).astype(np.int32), 1)
        if inner_scale is not None:
            inner = center + float(inner_scale) * (polygon - center)
            cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 0)
        selected = mask[vc, uc].astype(bool)
        return points_c[:, selected].T

    def depth_points_in_polygon(self, depth, polygon):
        return self.depth_points_in_region(
            depth,
            polygon,
            float(self.get_parameter('polygon_shrink').value))

    def fixed_panel_fallback_centroid(
            self, depth, corners, center_uv, fixed_approach_c):
        panel_points = self.depth_points_in_region(
            depth,
            corners,
            float(self.get_parameter('panel_outer_scale').value),
            float(self.get_parameter('panel_inner_scale').value))
        panel_min_points = int(self.get_parameter('panel_min_points').value)
        if panel_points is None or len(panel_points) < panel_min_points:
            return None, (
                'panel fallback has too few points: '
                f'{0 if panel_points is None else len(panel_points)}')

        panel_fit = fit_plane_ransac(
            panel_points,
            int(self.get_parameter('panel_ransac_iterations').value),
            float(self.get_parameter('panel_inlier_threshold_m').value))
        if panel_fit is None:
            return None, 'panel fallback RANSAC failed'
        min_ratio = float(self.get_parameter('panel_min_inlier_ratio').value)
        if panel_fit['inlier_ratio'] < min_ratio:
            return None, (
                f'panel fallback inlier ratio too low: '
                f"{panel_fit['inlier_ratio']:.2f} < {min_ratio:.2f}")
        max_rms = float(self.get_parameter('panel_max_rms_m').value)
        if panel_fit['rms'] > max_rms:
            return None, (
                f'panel fallback residual too high: '
                f"{panel_fit['rms'] * 1000.0:.1f}mm")

        panel_centroid = panel_fit['centroid']
        panel_normal = normalize(panel_fit['normal'])
        if panel_normal is None:
            return None, 'panel fallback normal is invalid'
        if float(np.dot(panel_normal, panel_centroid)) > 0.0:
            panel_normal = -panel_normal
        panel_approach = -panel_normal
        angle = math.degrees(math.acos(np.clip(
            float(np.dot(panel_approach, fixed_approach_c)), -1.0, 1.0)))
        max_angle = float(self.get_parameter(
            'panel_fixed_axis_max_angle_deg').value)
        if angle > max_angle:
            return None, (
                f'panel fallback/fixed-axis angle too large: '
                f'{angle:.1f}deg > {max_angle:.1f}deg')

        center_ray = ray_from_pixel(self.color_info, center_uv)
        panel_center = ray_plane_intersection(
            center_ray, panel_centroid, panel_normal)
        if panel_center is None:
            return None, 'center ray does not intersect panel fallback plane'
        offset = float(self.get_parameter(
            'panel_to_knob_center_offset_m').value)
        knob_center = panel_center - fixed_approach_c * offset
        self.get_logger().info(
            '[安装面回退] inliers=%d/%d(%.2f), rms=%.2fmm, '
            'axis_angle=%.1fdeg, center_offset=%.1fmm' % (
                panel_fit['inlier_count'], len(panel_points),
                panel_fit['inlier_ratio'], panel_fit['rms'] * 1000.0,
                angle, offset * 1000.0),
            throttle_duration_sec=1.0)
        return knob_center, 'ok'

    def localize_target(self, depth, detection, base_to_camera):
        xy = detection['keypoints']
        corners = xy[:4]
        geometric_center_2d = diagonal_intersection(corners)
        target_center_source = str(
            self.get_parameter('target_center_source').value).lower()
        center_uv = geometric_center_2d if target_center_source == 'geometric' else xy[4]
        points_c = self.depth_points_in_polygon(depth, corners)
        min_points = int(self.get_parameter('min_plane_points').value)
        if points_c is None or len(points_c) < min_points:
            return None, f'not enough valid depth points: {0 if points_c is None else len(points_c)}'

        raw_points_c = np.asarray(points_c, dtype=float)
        median = np.median(points_c, axis=0)
        distance = np.linalg.norm(points_c - median, axis=1)
        mad = np.median(np.abs(distance - np.median(distance))) + 1e-6
        points_c = points_c[distance < np.median(distance) + 3.5 * mad]
        if len(points_c) < min_points:
            return None, 'not enough inlier depth points after filtering'

        plane_source = str(self.get_parameter('plane_source').value).lower()
        lock_approach_axis = as_bool(
            self.get_parameter('lock_approach_axis_base').value)
        fixed_plane_depth_source = str(
            self.get_parameter('fixed_plane_depth_source').value).lower()
        resolved_depth_source = fixed_plane_depth_source
        if plane_source == 'pnp':
            long_size_model = float(
                self.get_parameter('pnp_long_size_m').value)
            short_size_model = float(
                self.get_parameter('pnp_short_size_m').value)
            half_long = 0.5 * long_size_model
            half_short = 0.5 * short_size_model
            object_points = np.asarray([
                [-half_long, -half_short, 0.0],
                [-half_long, half_short, 0.0],
                [half_long, half_short, 0.0],
                [half_long, -half_short, 0.0],
            ], dtype=np.float64)
            camera_matrix = np.asarray(
                self.color_info.k, dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(
                self.color_info.d, dtype=np.float64).reshape(-1, 1)
            try:
                result = cv2.solvePnPGeneric(
                    object_points,
                    np.asarray(corners, dtype=np.float64),
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_IPPE)
            except cv2.error as exc:
                return None, f'PnP failed: {exc}'
            if not result[0] or len(result[1]) == 0:
                return None, 'PnP did not return a valid pose'

            expected_approach_b = normalize(np.asarray(
                self.get_parameter('pnp_expected_approach_axis_base').value,
                dtype=float))
            if expected_approach_b is None:
                return None, 'invalid expected PnP approach axis'
            max_approach_deviation = float(
                self.get_parameter('pnp_max_approach_deviation_deg').value)
            continuity_weight = float(
                self.get_parameter('pnp_continuity_weight_px_per_deg').value)
            base_rotation_camera = base_to_camera[:3, :3]
            candidates = []
            for rvec, tvec in zip(result[1], result[2]):
                translation = np.asarray(tvec, dtype=float).reshape(3)
                if translation[2] <= 0.0:
                    continue
                projected, _ = cv2.projectPoints(
                    object_points, rvec, tvec, camera_matrix, distortion)
                projected = projected.reshape(-1, 2)
                reprojection_error = float(np.sqrt(np.mean(np.sum(
                    np.square(projected - corners), axis=1))))
                object_rotation, _ = cv2.Rodrigues(rvec)
                candidate_normal_c = normalize(object_rotation[:, 2])
                if candidate_normal_c is None:
                    continue
                if np.dot(candidate_normal_c, translation) > 0.0:
                    candidate_normal_c = -candidate_normal_c
                candidate_approach_b = normalize(
                    base_rotation_camera.dot(-candidate_normal_c))
                if candidate_approach_b is None:
                    continue
                expected_angle = math.degrees(math.acos(np.clip(
                    float(np.dot(candidate_approach_b, expected_approach_b)),
                    -1.0, 1.0)))
                if expected_angle > max_approach_deviation:
                    continue
                continuity_angle = 0.0
                if self.last_pnp_approach_b is not None:
                    continuity_angle = math.degrees(math.acos(np.clip(
                        float(np.dot(
                            candidate_approach_b,
                            self.last_pnp_approach_b)), -1.0, 1.0)))
                score = reprojection_error + continuity_weight * (
                    continuity_angle if self.last_pnp_approach_b is not None
                    else expected_angle)
                candidates.append((
                    score, reprojection_error, rvec, translation,
                    candidate_normal_c, candidate_approach_b,
                    expected_angle, continuity_angle))
            if not candidates:
                return None, (
                    'PnP has no solution within expected approach cone '
                    f'({max_approach_deviation:.1f}deg)')
            (_, reprojection_error, rvec, centroid, normal_c,
             selected_approach_b, expected_angle, continuity_angle) = min(
                candidates, key=lambda item: item[0])
            if reprojection_error > float(
                    self.get_parameter('pnp_max_reprojection_error_px').value):
                return None, (
                    f'PnP reprojection error too high: '
                    f'{reprojection_error:.1f}px')

            depth_center = np.median(points_c, axis=0)
            center_error = float(np.linalg.norm(depth_center - centroid))
            use_depth_anchored_plane = (
                lock_approach_axis and
                fixed_plane_depth_source in (
                    'depth_median', 'two_surface_midpoint'))
            if center_error > 0.060 and not use_depth_anchored_plane:
                return None, (
                    f'PnP/depth centers disagree by '
                    f'{center_error * 1000.0:.1f}mm')
            self.get_logger().info(
                '[PnP姿态] reprojection=%.2fpx, depth_delta=%.1fmm, '
                'expected_angle=%.1fdeg, continuity=%.1fdeg, '
                'normal=(%.3f, %.3f, %.3f)' % (
                    reprojection_error, center_error * 1000.0,
                    expected_angle, continuity_angle, *normal_c),
                throttle_duration_sec=1.0)
            self.last_pnp_approach_b = selected_approach_b
        elif plane_source == 'surrounding_panel':
            panel_points = self.depth_points_in_region(
                depth,
                corners,
                float(self.get_parameter('panel_outer_scale').value),
                float(self.get_parameter('panel_inner_scale').value))
            panel_min_points = int(self.get_parameter('panel_min_points').value)
            if panel_points is None or len(panel_points) < panel_min_points:
                return None, (
                    'not enough surrounding panel depth points: '
                    f'{0 if panel_points is None else len(panel_points)}')
            panel_fit = fit_plane_ransac(
                panel_points,
                int(self.get_parameter('panel_ransac_iterations').value),
                float(self.get_parameter('panel_inlier_threshold_m').value))
            if panel_fit is None:
                return None, 'surrounding panel RANSAC failed'
            if panel_fit['inlier_ratio'] < float(
                    self.get_parameter('panel_min_inlier_ratio').value):
                return None, (
                    'surrounding panel inlier ratio too low: '
                    f"{panel_fit['inlier_ratio']:.2f}")
            if panel_fit['rms'] > float(
                    self.get_parameter('panel_max_rms_m').value):
                return None, (
                    'surrounding panel residual too high: '
                    f"{panel_fit['rms'] * 1000.0:.1f}mm")

            panel_centroid = panel_fit['centroid']
            normal_c = panel_fit['normal']
            if np.dot(normal_c, panel_centroid) > 0.0:
                normal_c = -normal_c
            normal_c = normalize(normal_c)
            if normal_c is None:
                return None, 'invalid surrounding panel normal'

            view_axis = normalize(panel_centroid)
            approach_c = -normal_c
            view_angle = math.degrees(math.acos(np.clip(
                float(np.dot(approach_c, view_axis)), -1.0, 1.0)))
            if view_angle > float(
                    self.get_parameter('panel_max_view_angle_deg').value):
                return None, f'panel view angle too large: {view_angle:.1f}deg'

            signed_heights = (points_c - panel_centroid).dot(normal_c)
            knob_max_height = float(
                self.get_parameter('knob_max_height_m').value)
            plausible_heights = signed_heights[
                (signed_heights >= 0.0) &
                (signed_heights <= knob_max_height)]
            if len(plausible_heights) < min_points // 2:
                return None, 'not enough knob surface points above panel'
            knob_height = float(np.percentile(
                plausible_heights,
                float(self.get_parameter('knob_height_percentile').value)))
            knob_height += float(
                self.get_parameter('knob_surface_offset_m').value)
            if not (float(self.get_parameter('knob_min_height_m').value) <=
                    knob_height <= knob_max_height):
                return None, (
                    f'knob height out of range: {knob_height * 1000.0:.1f}mm')
            centroid = panel_centroid + normal_c * knob_height
            self.get_logger().info(
                '[安装面] inliers=%d/%d(%.2f), rms=%.2fmm, '
                'view=%.1fdeg, knob_height=%.1fmm' % (
                    panel_fit['inlier_count'], len(panel_points),
                    panel_fit['inlier_ratio'], panel_fit['rms'] * 1000.0,
                    view_angle, knob_height * 1000.0),
                throttle_duration_sec=1.0)
        else:
            centroid = points_c.mean(axis=0)
            _, _, vh = np.linalg.svd(points_c - centroid, full_matrices=False)
            normal_c = vh[-1]
            if np.dot(normal_c, centroid) > 0.0:
                normal_c = -normal_c
            normal_c = normalize(normal_c)
            if normal_c is None:
                return None, 'invalid knob plane normal'

        if lock_approach_axis:
            fixed_approach_b = normalize(np.asarray(
                self.get_parameter('fixed_approach_axis_base').value,
                dtype=float))
            if fixed_approach_b is None:
                return None, 'invalid fixed approach axis in base frame'
            fixed_approach_c = normalize(
                base_to_camera[:3, :3].T.dot(fixed_approach_b))
            if fixed_approach_c is None:
                return None, 'failed to transform fixed approach axis'
            # normal_c points from the target toward the camera; the tool
            # approaches in the opposite direction.
            normal_c = -fixed_approach_c
            if fixed_plane_depth_source == 'depth_median':
                centroid = np.median(points_c, axis=0)
            elif fixed_plane_depth_source == 'two_surface_midpoint':
                depths = raw_points_c.dot(fixed_approach_c)
                low, high = np.percentile(depths, [2.0, 98.0])
                depths = depths[(depths >= low) & (depths <= high)]
                if len(depths) < min_points:
                    return None, 'not enough depth points for two-surface fit'

                centers = np.asarray(
                    np.percentile(depths, [25.0, 75.0]), dtype=float)
                labels = np.zeros(len(depths), dtype=bool)
                for _ in range(12):
                    labels = (
                        np.abs(depths - centers[1]) <
                        np.abs(depths - centers[0]))
                    if not np.any(labels) or np.all(labels):
                        return None, 'two-surface depth clustering collapsed'
                    updated = np.asarray([
                        float(np.median(depths[~labels])),
                        float(np.median(depths[labels])),
                    ])
                    if np.max(np.abs(updated - centers)) < 1e-5:
                        centers = updated
                        break
                    centers = updated
                centers.sort()
                separation = float(centers[1] - centers[0])
                cluster_ratio = min(
                    float(np.mean(labels)), float(np.mean(~labels)))
                min_separation = float(self.get_parameter(
                    'fixed_plane_min_surface_separation_m').value)
                max_separation = float(self.get_parameter(
                    'fixed_plane_max_surface_separation_m').value)
                min_cluster_ratio = float(self.get_parameter(
                    'fixed_plane_min_cluster_ratio').value)
                depth_error = None
                if not (min_separation <= separation <= max_separation):
                    depth_error = (
                        f'two-surface separation invalid: '
                        f'{separation * 1000.0:.1f}mm')
                elif cluster_ratio < min_cluster_ratio:
                    depth_error = (
                        f'two-surface minority ratio too low: '
                        f'{cluster_ratio:.2f}')

                if depth_error is None:
                    plane_depth = float(np.mean(centers))
                    centroid = np.median(points_c, axis=0)
                    centroid = (
                        centroid + fixed_approach_c *
                        (plane_depth - float(np.dot(
                            centroid, fixed_approach_c))))
                    self.get_logger().info(
                        '[双表面深度] near=%.1fmm, far=%.1fmm, span=%.1fmm, '
                        'minority=%.2f' % (
                            centers[0] * 1000.0, centers[1] * 1000.0,
                            separation * 1000.0, cluster_ratio),
                        throttle_duration_sec=1.0)
                else:
                    single_surface_enabled = as_bool(self.get_parameter(
                        'fixed_plane_single_surface_fallback_enabled').value)
                    single_surface_max = float(self.get_parameter(
                        'single_surface_max_separation_m').value)
                    if single_surface_enabled and separation <= single_surface_max:
                        # Along the fixed +Y approach axis, the smaller cluster
                        # is the surface nearer the camera/robot.  The global
                        # median can be dominated by the rear surface.
                        front_depth = float(centers[0])
                        center_offset = float(self.get_parameter(
                            'knob_front_to_center_offset_m').value)
                        extra_offset = float(self.get_parameter(
                            'single_surface_extra_depth_offset_m').value)
                        plane_depth = front_depth + center_offset + extra_offset
                        centroid = np.median(points_c, axis=0)
                        centroid = (
                            centroid + fixed_approach_c *
                            (plane_depth - float(np.dot(
                                centroid, fixed_approach_c))))
                        resolved_depth_source = 'single_surface_offset'
                        self.get_logger().info(
                            '[单表面回退] span=%.1fmm, front=%.1fmm, '
                            'center_offset=%.1fmm, extra_offset=%.1fmm' % (
                                separation * 1000.0, front_depth * 1000.0,
                                center_offset * 1000.0,
                                extra_offset * 1000.0),
                            throttle_duration_sec=1.0)
                    else:
                        if not as_bool(self.get_parameter(
                                'fixed_plane_panel_fallback_enabled').value):
                            return None, depth_error
                        centroid, panel_error = self.fixed_panel_fallback_centroid(
                            depth, corners, center_uv, fixed_approach_c)
                        if centroid is None:
                            return None, f'{depth_error}; {panel_error}'
                        resolved_depth_source = 'panel_offset'
                        self.get_logger().info(
                            f'[深度回退] {depth_error}，改用外围安装面',
                            throttle_duration_sec=1.0)
            elif fixed_plane_depth_source != 'pnp':
                return None, (
                    'fixed_plane_depth_source must be two_surface_midpoint, '
                    'depth_median or pnp')
            centroid = (
                centroid + fixed_approach_c *
                float(self.get_parameter(
                    'fixed_plane_depth_offset_m').value))
            self.get_logger().info(
                '[固定法向] approach_base=(%.3f, %.3f, %.3f), depth=%s' %
                (*fixed_approach_b, resolved_depth_source),
                throttle_duration_sec=2.0)

        points_on_plane = []
        for uv in corners:
            ray = ray_from_pixel(self.color_info, uv)
            point = ray_plane_intersection(ray, centroid, normal_c)
            if point is None:
                return None, 'keypoint ray does not intersect fitted plane'
            points_on_plane.append(point)
        points_on_plane = np.asarray(points_on_plane)
        corner_3d = points_on_plane
        geometric_center_c = ray_plane_intersection(
            ray_from_pixel(self.color_info, geometric_center_2d), centroid, normal_c)
        model_center_c = ray_plane_intersection(
            ray_from_pixel(self.color_info, xy[4]), centroid, normal_c)
        if geometric_center_c is None or model_center_c is None:
            return None, 'center ray does not intersect fitted plane'
        center_delta = float(np.linalg.norm(model_center_c - geometric_center_c))
        blend_limit = float(self.get_parameter(
            'center_blend_max_difference_m').value)
        reject_limit = float(self.get_parameter(
            'center_reject_difference_m').value)
        if center_delta > reject_limit:
            return None, (
                'YOLO/geometric center disagreement too large: '
                f'{center_delta * 1000.0:.1f}mm')
        if center_delta > blend_limit:
            return None, (
                'YOLO/geometric center not stable yet: '
                f'{center_delta * 1000.0:.1f}mm')
        center_c = 0.5 * (model_center_c + geometric_center_c)
        if center_delta > 1e-3:
            self.get_logger().info(
                '[视觉检查] YOLO center 与几何中心相差 %.1fmm，使用两者平均' %
                (center_delta * 1000.0),
                throttle_duration_sec=1.0)

        long_end_1_2d = midpoint(corners[0], corners[1])
        long_end_2_2d = midpoint(corners[2], corners[3])
        side_1_2d = midpoint(corners[1], corners[2])
        side_2_2d = midpoint(corners[3], corners[0])
        key_plane_points = []
        for uv in (long_end_1_2d, long_end_2_2d, side_1_2d, side_2_2d):
            ray = ray_from_pixel(self.color_info, uv)
            point = ray_plane_intersection(ray, centroid, normal_c)
            if point is None:
                return None, 'axis midpoint ray does not intersect fitted plane'
            key_plane_points.append(point)
        long_end_1, long_end_2, side_1, side_2 = key_plane_points
        long_axis_c = normalize(long_end_2 - long_end_1)
        close_axis_c = normalize(side_2 - side_1)
        if long_axis_c is None or close_axis_c is None:
            return None, 'invalid 3D axes'
        close_axis_c = normalize(close_axis_c - normal_c * np.dot(close_axis_c, normal_c))
        long_axis_c = normalize(long_axis_c - normal_c * np.dot(long_axis_c, normal_c))
        long_axis_c = normalize(long_axis_c - close_axis_c * np.dot(long_axis_c, close_axis_c))
        if long_axis_c is None:
            long_axis_c = normalize(np.cross(close_axis_c, normal_c))
        if close_axis_c is None or long_axis_c is None:
            return None, 'invalid orthogonalized axes'

        long_size = float(np.linalg.norm(long_end_2 - long_end_1))
        short_size = float(np.linalg.norm(side_2 - side_1))
        size_warnings = []
        if not (float(self.get_parameter('min_long_size_m').value) <= long_size <=
                float(self.get_parameter('max_long_size_m').value)):
            size_warnings.append(f'long size out of range: {long_size * 1000.0:.1f}mm')
        if not (float(self.get_parameter('min_short_size_m').value) <= short_size <=
                float(self.get_parameter('max_short_size_m').value)):
            size_warnings.append(f'short size out of range: {short_size * 1000.0:.1f}mm')
        if size_warnings and as_bool(self.get_parameter('reject_size_out_of_range').value):
            return None, '; '.join(size_warnings)

        rotation_c = np.column_stack((close_axis_c, long_axis_c, -normal_c))
        if np.linalg.det(rotation_c) < 0.0:
            rotation_c[:, 1] *= -1.0
        orientation_offset = matrix_from_xyz_rpy(
            [0.0, 0.0, 0.0],
            self.get_parameter('grasp_orientation_offset_rpy').value)
        rotation_c = rotation_c.dot(orientation_offset[:3, :3])

        rotation_b = base_to_camera[:3, :3].dot(rotation_c)
        center_b = base_to_camera.dot(np.r_[center_c, 1.0])[:3]
        target_offset = np.asarray(
            self.get_parameter('target_position_offset_m').value,
            dtype=float)
        if target_offset.shape == (3,):
            center_b = center_b + rotation_b.dot(target_offset)
        normal_b = base_to_camera[:3, :3].dot(normal_c)
        close_axis_b = rotation_b[:, 0]
        long_axis_b = rotation_b[:, 1]
        approach_axis_b = normalize(
            rotation_b[:, 2] *
            float(self.get_parameter('approach_axis_sign').value))
        if approach_axis_b is None:
            return None, 'invalid approach axis'
        self.get_logger().info(
            '[目标位姿] center=(%.4f, %.4f, %.4f), approach=(%.3f, %.3f, %.3f), '
            'close=(%.3f, %.3f, %.3f), size=(%.1f x %.1f)mm' %
            (*center_b, *approach_axis_b, *close_axis_b,
             long_size * 1000.0, short_size * 1000.0),
            throttle_duration_sec=1.0)

        target_matrix = np.eye(4)
        target_matrix[:3, :3] = rotation_b
        target_matrix[:3, 3] = center_b
        pregrasp_matrix = target_matrix.copy()
        pregrasp_matrix[:3, 3] = (
            center_b - approach_axis_b *
            float(self.get_parameter('pregrasp_distance_m').value))

        return {
            'position': center_b,
            'rotation': rotation_b,
            'matrix': target_matrix,
            'pregrasp_matrix': pregrasp_matrix,
            'normal': normal_b,
            'close_axis': close_axis_b,
            'long_axis': long_axis_b,
            'approach_axis': approach_axis_b,
            'depth_source': resolved_depth_source,
            'long_size': long_size,
            'short_size': short_size,
            'stamp': self.get_clock().now(),
            'angle': signed_angle_180(detection['keypoints'][2] - detection['keypoints'][0]),
        }, 'ok' if not size_warnings else '; '.join(size_warnings)

    def update_stability(self, target):
        if (self.history and
                self.history[-1].get('depth_source') != target.get('depth_source')):
            self.get_logger().info(
                '[稳定性] 深度来源从 %s 切换为 %s，重新累计稳定帧' % (
                    self.history[-1].get('depth_source'),
                    target.get('depth_source')),
                throttle_duration_sec=1.0)
            self.history.clear()
        self.history.append(target)
        count = int(self.get_parameter('stable_frames').value)
        if len(self.history) < count:
            return None
        positions = np.asarray([item['position'] for item in self.history])
        if np.max(np.std(positions, axis=0)) > float(self.get_parameter('stable_position_std_m').value):
            return None
        angles = np.unwrap(np.deg2rad([item['angle'] * 2.0 for item in self.history])) * 0.5
        if math.degrees(float(np.std(angles))) > float(self.get_parameter('stable_angle_std_deg').value):
            return None
        normal_spread = axis_spread_deg(
            [item['normal'] for item in self.history], directed=True)
        close_spread = axis_spread_deg(
            [item['close_axis'] for item in self.history], directed=True)
        long_spread = axis_spread_deg(
            [item['long_axis'] for item in self.history], directed=True)
        if normal_spread > float(
                self.get_parameter('stable_normal_spread_deg').value):
            return None
        if max(close_spread, long_spread) > float(
                self.get_parameter('stable_axis_spread_deg').value):
            return None
        stable = dict(self.history[-1])
        stable['position'] = positions.mean(axis=0)

        # Position stability alone is not enough for a side grasp.  PnP can
        # jitter a few degrees between otherwise valid frames, so using the
        # last frame makes the flange error change direction from run to run.
        # Fuse the accepted axes and rebuild an orthonormal right-handed frame.
        approach_axis = normalize(np.mean(
            [item['approach_axis'] for item in self.history], axis=0))
        close_axis = normalize(np.mean(
            [item['close_axis'] for item in self.history], axis=0))
        normal_axis = normalize(np.mean(
            [item['normal'] for item in self.history], axis=0))
        if approach_axis is None or close_axis is None or normal_axis is None:
            return None

        axis_sign = float(self.get_parameter('approach_axis_sign').value)
        rotation_z = normalize(approach_axis * axis_sign)
        if rotation_z is None:
            return None
        close_axis = normalize(
            close_axis - rotation_z * float(np.dot(close_axis, rotation_z)))
        if close_axis is None:
            return None
        long_axis = normalize(np.cross(rotation_z, close_axis))
        if long_axis is None:
            return None

        rotation = np.column_stack((close_axis, long_axis, rotation_z))
        stable['rotation'] = rotation
        stable['normal'] = normal_axis
        stable['close_axis'] = close_axis
        stable['long_axis'] = long_axis
        stable['approach_axis'] = approach_axis
        stable['matrix'] = np.array(stable['matrix'])
        stable['matrix'][:3, :3] = rotation
        stable['matrix'][:3, 3] = stable['position']
        stable['pregrasp_matrix'] = np.array(stable['pregrasp_matrix'])
        stable['pregrasp_matrix'][:3, :3] = rotation
        stable['pregrasp_matrix'][:3, 3] = (
            stable['position'] - stable['approach_axis'] *
            float(self.get_parameter('pregrasp_distance_m').value))
        stable['stamp'] = self.get_clock().now()
        return stable

    def publish_pose(self, target):
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose = pose_from_matrix(target['matrix'])
        self.pose_pub.publish(msg)

    def clear_markers(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.base_frame
        marker.action = Marker.DELETEALL
        markers = MarkerArray()
        markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def publish_markers(self, target, stable):
        markers = MarkerArray()
        header_stamp = self.get_clock().now().to_msg()
        for index, (name, matrix, color) in enumerate([
            ('grasp', target['matrix'], (0.1, 1.0, 0.2)),
            ('pregrasp', target['pregrasp_matrix'], (1.0, 0.7, 0.1)),
        ]):
            marker = Marker()
            marker.header.stamp = header_stamp
            marker.header.frame_id = self.base_frame
            marker.ns = 'eco65_knob_' + name
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose_from_matrix(matrix)
            marker.scale.x = 0.08
            marker.scale.y = 0.012
            marker.scale.z = 0.012
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0 if stable else 0.55
            markers.markers.append(marker)

        axis_specs = [
            ('close_axis', target['close_axis'], (1.0, 0.0, 1.0), 10),
            ('long_axis', target['long_axis'], (0.0, 0.9, 1.0), 11),
            ('approach_axis', target['approach_axis'], (1.0, 0.2, 0.1), 12),
        ]
        for ns, axis, color, marker_id in axis_specs:
            marker = Marker()
            marker.header.stamp = header_stamp
            marker.header.frame_id = self.base_frame
            marker.ns = 'eco65_knob_' + ns
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = float(target['position'][0])
            marker.pose.position.y = float(target['position'][1])
            marker.pose.position.z = float(target['position'][2])
            axis = normalize(axis, [1.0, 0.0, 0.0])
            rot = np.eye(4)
            rot[:3, 0] = axis
            helper = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(helper, axis))) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            rot[:3, 1] = normalize(np.cross(helper, axis))
            rot[:3, 2] = normalize(np.cross(axis, rot[:3, 1]))
            marker.pose.orientation = pose_from_matrix(rot).orientation
            marker.scale.x = 0.055
            marker.scale.y = 0.006
            marker.scale.z = 0.006
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            markers.markers.append(marker)

        self.marker_pub.publish(markers)

    def draw_detection(self, image, detection, localized, reason):
        box = detection['box'].astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 80, 0), 2)
        points = detection['keypoints']
        for point, color, name, conf in zip(points, KEYPOINT_COLORS, KEYPOINT_NAMES, detection['keypoint_conf']):
            location = tuple(int(round(v)) for v in point)
            cv2.circle(image, location, 5, color, -1, cv2.LINE_AA)
            cv2.putText(image, f'{name}:{conf:.2f}', (location[0] + 5, location[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
        corners = points[:4].astype(np.int32)
        cv2.polylines(image, [corners], True, (255, 150, 0), 2, cv2.LINE_AA)
        long_end_1 = midpoint(points[0], points[1])
        long_end_2 = midpoint(points[2], points[3])
        side_1 = midpoint(points[1], points[2])
        side_2 = midpoint(points[3], points[0])
        cv2.line(image, tuple(long_end_1.astype(int)), tuple(long_end_2.astype(int)),
                 (0, 220, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(image, tuple(side_1.astype(int)), tuple(side_2.astype(int)),
                        (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.12)
        text = 'localized' if localized else reason
        color = (0, 255, 0) if localized else (0, 0, 255)
        cv2.putText(image, text, (max(5, box[0]), max(24, box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    def publish_debug(self, image):
        if self.debug_pub.get_subscription_count() == 0:
            return
        image = np.ascontiguousarray(image, dtype=np.uint8)
        msg = Image()
        msg.header = self.color_msg.header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * image.shape[2])
        msg.data = image.tobytes()
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Eco65KnobDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
