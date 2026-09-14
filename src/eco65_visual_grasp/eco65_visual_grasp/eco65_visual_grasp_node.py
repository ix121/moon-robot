import collections
from contextlib import contextmanager
import json
import math
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.exceptions import ParameterUninitializedException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Armcurrentstatus, Movej, Movejp, Movel, Sixforce
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:
    import serial
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    serial = None

try:
    from orbbec_camera_msgs.msg import Extrinsics
except ImportError:  # pragma: no cover
    Extrinsics = None


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def matrix_from_xyz_rpy(xyz, rpy):
    cx, sx = math.cos(rpy[0]), math.sin(rpy[0])
    cy, sy = math.cos(rpy[1]), math.sin(rpy[1])
    cz, sz = math.cos(rpy[2]), math.sin(rpy[2])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    result = np.eye(4)
    result[:3, :3] = rz.dot(ry).dot(rx)
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


def matrix_from_quaternion_translation(q, t):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        rot = np.eye(3)
    else:
        s = 2.0 / n
        rot = np.array([
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ])
    result = np.eye(4)
    result[:3, :3] = rot
    result[:3, 3] = np.asarray(t, dtype=float)
    return result


def matrix_from_transform(transform):
    q = transform.rotation
    t = transform.translation
    return matrix_from_quaternion_translation(
        [q.x, q.y, q.z, q.w], [t.x, t.y, t.z])


def quaternion_from_matrix(matrix):
    rotation = matrix[:3, :3]
    trace = np.trace(rotation)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def pose_from_matrix(matrix):
    pose = Pose()
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    qx, qy, qz, qw = quaternion_from_matrix(matrix)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


class Eco65VisualGrasp(Node):
    def __init__(self):
        super().__init__('eco65_visual_grasp')
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.motion_lock = threading.Lock()
        self.cb_group = ReentrantCallbackGroup()
        self.cable_insertion_active = threading.Event()

        self._declare_parameters()
        self.base_frame = self.p('base_frame')
        self.flange_frame = self.p('flange_frame')
        self.execute_motion = as_bool(self.p('execute_motion'))
        self.joint_names = list(self.p('joint_names'))
        self.tool_transform = matrix_from_xyz_rpy(self.p('tool_xyz'), self.p('tool_rpy'))
        self.flange_to_camera = self.load_handeye(self.p('handeye_file'))

        self.color_info = None
        self.depth_info = None
        self.depth_to_color = None
        self.latest_color = None
        self.latest_depth = None
        self.latest_joints = {}
        self.latest_joint_time = self.get_clock().now()
        self.histories = collections.defaultdict(
            lambda: collections.deque(maxlen=int(self.p('stable_frames'))))
        self.stable_targets = {}
        self.target_collection_enabled = True
        self.validated_target = None
        self.last_cable_preinsert_tip = None
        self.fz_samples = collections.deque(maxlen=500)
        self.fz_sequence = 0
        self.fz_last_callback_stamp = None
        self.fz_max_gap_during_insertion = 0.0
        self.arm_status = None
        self.arm_status_sequence = 0
        self.controller_pose_matrix = None
        self.controller_pose_stamp = None

        dictionary_name = self.p('aruco_dictionary')
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._make_subscriptions()
        self.target_pub = self.create_publisher(PoseStamped, '~/target_pose', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)

        self.movej_pub = self.create_publisher(Movej, self.p('movej_topic'), 10)
        self.movejp_pub = self.create_publisher(Movejp, self.p('movej_p_topic'), 10)
        self.movel_pub = self.create_publisher(Movel, self.p('movel_topic'), 10)
        self.move_stop_pub = self.create_publisher(
            Empty, self.p('move_stop_topic'), 10)
        self.movej_result = None
        self.movejp_result = None
        self.movel_result = None
        self.create_subscription(
            Bool, self.p('movej_result_topic'), self._movej_result_cb, 10,
            callback_group=self.cb_group)
        self.create_subscription(
            Bool, self.p('movej_p_result_topic'), self._movejp_result_cb, 10,
            callback_group=self.cb_group)
        self.create_subscription(
            Bool, self.p('movel_result_topic'), self._movel_result_cb, 10,
            callback_group=self.cb_group)

        self.observation_client = self.create_client(
            Trigger, self.p('observation_service'), callback_group=self.cb_group)
        self.compliance_zero_client = self.create_client(
            Trigger, self.p('compliance_zero_service'), callback_group=self.cb_group)
        self.compliance_start_client = self.create_client(
            Trigger, self.p('compliance_start_service'), callback_group=self.cb_group)

        self.create_service(Trigger, '~/preview', self.preview_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/move_to_observation', self.observation_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/open_gripper', self.open_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/close_gripper', self.close_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/search_target', self.search_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/move_to_pregrasp', self.move_pregrasp_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/complete_from_pregrasp', self.complete_from_pregrasp_cb, callback_group=self.cb_group)
        self.create_service(Trigger, '~/grasp_nearest', self.grasp_cb, callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/preview_cable_preinsert',
            self.preview_cable_preinsert_cb, callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/move_to_cable_preinsert',
            self.move_to_cable_preinsert_cb, callback_group=self.cb_group)
        self.create_service(
            Trigger, '~/insert_cable',
            self.insert_cable_cb, callback_group=self.cb_group)
        self.create_service(
            Trigger,
            '~/grasp_object_and_move_to_place',
            self.grasp_object_and_move_to_place_cb,
            callback_group=self.cb_group)

        self.create_timer(0.10, self.process_frame)
        self.get_logger().info(
            f'ECO65 visual grasp ready. execute_motion={self.execute_motion}, '
            f'base={self.base_frame}, flange={self.flange_frame}')

    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text, throttle_duration_sec=2.0)

    def _declare_parameters(self):
        self.param_defaults = {
            'base_frame': 'baselink',
            'flange_frame': 'Link6',
            'handeye_file': '/home/proton/eco65_grasp_ws/handeye_config.json',
            'tool_xyz': [0.0, 0.0, 0.144],
            'tool_rpy': [0.0, 0.0, 0.0],
            'color_image_topic': '/camera/color/image_raw',
            'color_info_topic': '/camera/color/camera_info',
            'depth_image_topic': '/camera/depth/image_raw',
            'depth_info_topic': '/camera/depth/camera_info',
            'depth_to_color_topic': '/camera/depth_to_color',
            'depth_registered_to_color': False,
            'aruco_dictionary': 'DICT_6X6_50',
            'marker_ids': [],
            'polygon_shrink': 0.72,
            'depth_min_m': 0.08,
            'depth_max_m': 1.20,
            'depth_stride': 2,
            'min_plane_points': 80,
            'max_surface_tilt_deg': 75.0,
            'workspace_min': [-0.55, -0.55, -0.20],
            'workspace_max': [0.75, 0.55, 0.65],
            'table_height_m': -0.20,
            'min_object_height_m': 0.015,
            'enable_table_filter': False,
            'stable_frames': 5,
            'stable_position_std_m': 0.006,
            'stable_yaw_std_deg': 6.0,
            'target_timeout_s': 1.0,
            'object_marker_id': 6,
            'place_marker_id': 0,
            'place_target_wait_s': 4.0,
            'place_above_height_m': 0.040,
            'place_xy_offset_m': [0.0, 0.0],
            'place_yaw_offset_deg': 0.0,
            'cable_socket_marker_id': 1,
            'cable_marker_to_preinsert_offset_base_m': [0.0, -0.21, -0.36],
            'cable_tool_tip_axis_base': [0.0, 1.0, 0.0],
            'cable_tool_x_axis_base': [0.0, 0.0, -1.0],
            'cable_insert_use_tool_tip_axis': True,
            'cable_insert_axis_base': [0.0, 1.0, 0.0],
            'cable_insert_tool_axis_sign': 1.0,
            'cable_insert_axis_min_alignment': 0.80,
            'cable_target_wait_s': 4.0,
            'cable_preinsert_speed': 6,
            'cable_preinsert_timeout_s': 60.0,
            'cable_preinsert_reached_position_tolerance_m': 0.015,
            'cable_preinsert_reached_orientation_tolerance_deg': 8.0,
            'cable_max_preinsert_move_m': 0.80,
            'cable_insert_distance_m': 0.05,
            'cable_insert_step_m': 0.01,
            'cable_insert_speed': 3,
            'cable_insert_force_threshold_n': 20.0,
            'cable_insert_force_trigger_samples': 1,
            'cable_insert_start_tolerance_m': 0.03,
            'cable_insert_position_tolerance_m': 0.004,
            'cable_insert_max_overshoot_m': 0.005,
            'cable_insert_baseline_window_s': 0.30,
            'cable_insert_baseline_min_samples': 20,
            'force_sample_timeout_s': 0.20,
            'force_first_sample_timeout_s': 1.00,
            'sixforce_topic': '/rm_driver/udp_six_zero_force',
            'arm_status_topic': '/rm_driver/udp_arm_current_status',
            'controller_pose_topic': '/rm_driver/udp_arm_position',
            'controller_pose_timeout_s': 0.20,
            'cable_marker_max_surface_tilt_deg': 95.0,
            'cable_marker_workspace_min': [-0.75, -0.75, -0.20],
            'cable_marker_workspace_max': [0.90, 0.90, 1.30],
            'observation_service': '/eco65_observation/move_to_observation',
            'joint_states_topic': '/joint_states',
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'enable_base_scan': True,
            'base_scan_joint_name': 'joint1',
            'base_scan_min_rad': -1.20,
            'base_scan_max_rad': 1.20,
            'base_scan_step_rad': 0.25,
            'base_scan_initial_direction': 1.0,
            'base_scan_speed': 10,
            'base_scan_settle_s': 2.0,
            'base_scan_step_dwell_s': 2.0,
            'base_scan_timeout_s': 90.0,
            'grasp_yaw_offset_deg': -90.0,
            'pregrasp_height_m': 0.050,
            'grasp_below_surface_m': 0.020,
            'lift_height_m': 0.050,
            'execute_motion': False,
            'movej_p_topic': '/rm_driver/movej_p_cmd',
            'movej_p_result_topic': '/rm_driver/movej_p_result',
            'movel_topic': '/rm_driver/movel_cmd',
            'movel_result_topic': '/rm_driver/movel_result',
            'move_stop_topic': '/rm_driver/move_stop_cmd',
            'movej_topic': '/rm_driver/movej_cmd',
            'movej_result_topic': '/rm_driver/movej_result',
            'pose_speed': 10,
            'linear_speed': 5,
            'joint_speed': 10,
            'command_timeout_s': 20.0,
            'gripper_port': '/dev/ttyACM0',
            'gripper_baud': 115200,
            'gripper_timeout_s': 1.0,
            'gripper_open_hex': '7b01020020492000c8f97d',
            'gripper_close_hex': '7b01020120492000c8f87d',
            'gripper_motion_wait_s': 1.0,
            'release_after_lift': True,
            'release_wait_s': 3.0,
            'start_compliance_after_lift': False,
            'compliance_settle_before_zero_s': 1.5,
            'compliance_zero_service': '/sixforce_compliance/zero_bias',
            'compliance_start_service': '/sixforce_compliance/start',
            'compliance_service_timeout_s': 5.0,
        }
        for name, value in self.param_defaults.items():
            self.declare_parameter(name, value)

    def p(self, name):
        try:
            return self.get_parameter(name).value
        except ParameterUninitializedException:
            return self.param_defaults[name]

    def load_handeye(self, path):
        with open(path, 'r') as stream:
            data = json.load(stream)
        if 'H' in data:
            matrix = np.asarray(data['H'], dtype=float)
        else:
            transform = data['transform']
            t = transform['translation_m']
            q = transform['quaternion_xyzw']
            matrix = matrix_from_quaternion_translation(q, [t['x'], t['y'], t['z']])
        if matrix.shape != (4, 4):
            raise RuntimeError('handeye matrix must be 4x4')
        self.get_logger().info(
            'Loaded flange_T_camera from %s; xyz=[%.4f %.4f %.4f]' %
            (path, matrix[0, 3], matrix[1, 3], matrix[2, 3]))
        return matrix

    def _make_subscriptions(self):
        self.create_subscription(Image, self.p('color_image_topic'), self.color_cb, 10)
        self.create_subscription(Image, self.p('depth_image_topic'), self.depth_cb, 10)
        self.create_subscription(CameraInfo, self.p('color_info_topic'), self.color_info_cb, 10)
        self.create_subscription(CameraInfo, self.p('depth_info_topic'), self.depth_info_cb, 10)
        self.create_subscription(JointState, self.p('joint_states_topic'), self.joint_cb, 10)
        self.create_subscription(
            Sixforce, self.p('sixforce_topic'), self.sixforce_cb, 10,
            callback_group=self.cb_group)
        self.create_subscription(
            Armcurrentstatus, self.p('arm_status_topic'), self.arm_status_cb, 10,
            callback_group=self.cb_group)
        self.create_subscription(
            Pose, self.p('controller_pose_topic'), self.controller_pose_cb, 10,
            callback_group=self.cb_group)
        if not as_bool(self.p('depth_registered_to_color')) and Extrinsics is not None:
            extrinsics_qos = QoSProfile(depth=1)
            extrinsics_qos.reliability = ReliabilityPolicy.RELIABLE
            extrinsics_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(
                Extrinsics,
                self.p('depth_to_color_topic'),
                self.extrinsics_cb,
                extrinsics_qos)
        elif not as_bool(self.p('depth_registered_to_color')):
            self.get_logger().warn(
                'orbbec_camera_msgs/Extrinsics is not importable; depth_to_color will be unavailable.')

    def color_cb(self, msg):
        if self.cable_insertion_active.is_set():
            return
        with self.lock:
            self.latest_color = msg

    def depth_cb(self, msg):
        if self.cable_insertion_active.is_set():
            return
        with self.lock:
            self.latest_depth = msg

    def color_info_cb(self, msg):
        if self.cable_insertion_active.is_set():
            return
        self.color_info = msg

    def depth_info_cb(self, msg):
        if self.cable_insertion_active.is_set():
            return
        self.depth_info = msg

    def extrinsics_cb(self, msg):
        if self.cable_insertion_active.is_set():
            return
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(msg.rotation, dtype=float).reshape(3, 3)
        matrix[:3, 3] = np.asarray(msg.translation, dtype=float)
        self.depth_to_color = matrix

    def joint_cb(self, msg):
        with self.lock:
            self.latest_joints = dict(zip(msg.name, msg.position))
            self.latest_joint_time = self.get_clock().now()

    def sixforce_cb(self, msg):
        stamp = time.monotonic()
        if (self.cable_insertion_active.is_set() and
                self.fz_last_callback_stamp is not None):
            self.fz_max_gap_during_insertion = max(
                self.fz_max_gap_during_insertion,
                stamp - self.fz_last_callback_stamp)
        self.fz_last_callback_stamp = stamp
        self.fz_sequence += 1
        self.fz_samples.append(
            (stamp, float(msg.force_fz), self.fz_sequence))

    def arm_status_cb(self, msg):
        self.arm_status = int(msg.arm_current_status)
        self.arm_status_sequence += 1

    def controller_pose_cb(self, msg):
        matrix = matrix_from_quaternion_translation(
            [msg.orientation.x, msg.orientation.y,
             msg.orientation.z, msg.orientation.w],
            [msg.position.x, msg.position.y, msg.position.z])
        with self.lock:
            self.controller_pose_matrix = matrix
            self.controller_pose_stamp = time.monotonic()

    def _movej_result_cb(self, msg):
        self.movej_result = bool(msg.data)

    def _movejp_result_cb(self, msg):
        self.movejp_result = bool(msg.data)

    def _movel_result_cb(self, msg):
        self.movel_result = bool(msg.data)

    def lookup_base_to_flange(self):
        tf_msg = self.tf_buffer.lookup_transform(
            self.base_frame, self.flange_frame, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5))
        return matrix_from_transform(tf_msg.transform)

    def process_frame(self):
        # Vision processing can take longer than one force-sensor period. Keep
        # the executor available to supervise force while inserting the cable.
        if self.cable_insertion_active.is_set():
            return
        with self.lock:
            color_msg = self.latest_color
            depth_msg = self.latest_depth
        if color_msg is None or depth_msg is None or self.color_info is None or self.depth_info is None:
            missing = []
            if color_msg is None:
                missing.append('color_image')
            if depth_msg is None:
                missing.append('depth_image')
            if self.color_info is None:
                missing.append('color_info')
            if self.depth_info is None:
                missing.append('depth_info')
            self.set_status('waiting for camera images/info: ' + ', '.join(missing))
            return
        if not as_bool(self.p('depth_registered_to_color')) and self.depth_to_color is None:
            self.set_status('waiting for ' + self.p('depth_to_color_topic'))
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            base_to_camera = self.lookup_base_to_flange().dot(self.flange_to_camera)
        except (TransformException, Exception) as exc:
            self.set_status(f'perception input unavailable: {exc}')
            return

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        debug = color.copy()
        if ids is None:
            self.set_status('no ArUco marker detected')
        else:
            allowed = set(int(v) for v in self.p('marker_ids'))
            valid_count = 0
            rejected_count = 0
            reject_reason = ''
            marker_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
            for marker_corners, marker_id_value in zip(corners, marker_ids):
                marker_id = int(marker_id_value)
                if allowed and marker_id not in allowed:
                    rejected_count += 1
                    reject_reason = f'marker {marker_id} not in marker_ids'
                    continue
                points = self.depth_points_in_marker(depth, marker_corners)
                target, target_error = self.fit_target(
                    points, marker_corners, base_to_camera, marker_id)
                if target is None:
                    rejected_count += 1
                    reject_reason = f'marker {marker_id} rejected: {target_error}'
                    continue
                valid_count += 1
                history = self.histories[marker_id]
                history.append((target['position'], target['yaw'], target['rotation'], self.get_clock().now()))
                stable = self.evaluate_stability(history)
                if stable is not None:
                    with self.lock:
                        self.stable_targets[marker_id] = stable
                    self.publish_target(marker_id, stable)
                    self.set_status(f'stable marker {marker_id} ready')
                else:
                    self.set_status(
                        f'marker {marker_id} detected; waiting for stability '
                        f'({len(history)}/{int(self.p("stable_frames"))})')
                cv2.polylines(debug, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(debug, f'id={marker_id}', tuple(marker_corners.reshape(4, 2)[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if valid_count == 0 and rejected_count > 0:
                self.set_status(reject_reason)
        self.publish_debug(color_msg, debug)

    def depth_points_in_marker(self, depth, corners):
        if self.depth_info is None or self.color_info is None:
            return None
        if not as_bool(self.p('depth_registered_to_color')) and self.depth_to_color is None:
            return None
        if depth.dtype == np.uint16:
            z_image = depth.astype(np.float32) * 0.001
        else:
            z_image = depth.astype(np.float32)
        stride = max(1, int(self.p('depth_stride')))
        rows, cols = np.mgrid[0:z_image.shape[0]:stride, 0:z_image.shape[1]:stride]
        z = z_image[::stride, ::stride].reshape(-1)
        u = cols.reshape(-1).astype(np.float32)
        v = rows.reshape(-1).astype(np.float32)
        valid = np.isfinite(z)
        valid &= z >= float(self.p('depth_min_m'))
        valid &= z <= float(self.p('depth_max_m'))
        u, v, z = u[valid], v[valid], z[valid]
        if z.size == 0:
            return None

        if as_bool(self.p('depth_registered_to_color')):
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

        polygon = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        center = polygon.mean(axis=0)
        polygon = center + float(self.p('polygon_shrink')) * (polygon - center)
        mask = np.zeros((self.color_info.height, self.color_info.width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
        selected = mask[vc, uc].astype(bool)
        return points_c[:, selected].T

    def fit_target(self, points_c, corners, base_to_camera, marker_id=None):
        min_points = int(self.p('min_plane_points'))
        if points_c is None or len(points_c) < min_points:
            count = 0 if points_c is None else len(points_c)
            return None, f'not enough depth points ({count}/{min_points})'
        median = np.median(points_c, axis=0)
        distance = np.linalg.norm(points_c - median, axis=1)
        mad = np.median(np.abs(distance - np.median(distance))) + 1e-6
        points_c = points_c[distance < np.median(distance) + 3.5 * mad]
        if len(points_c) < min_points:
            return None, (
                f'not enough filtered depth points ({len(points_c)}/{min_points})')

        centroid = points_c.mean(axis=0)
        _, _, vh = np.linalg.svd(points_c - centroid, full_matrices=False)
        normal_c = vh[-1]
        if np.dot(normal_c, centroid) > 0:
            normal_c = -normal_c
        center_b = base_to_camera.dot(np.r_[centroid, 1.0])[:3]
        normal_b = base_to_camera[:3, :3].dot(normal_c)
        upward = normal_b if normal_b[2] >= 0 else -normal_b
        tilt = math.degrees(math.acos(np.clip(upward[2] / max(np.linalg.norm(upward), 1e-9), -1.0, 1.0)))
        is_cable_marker = (
            marker_id is not None and
            int(marker_id) == int(self.p('cable_socket_marker_id')))
        max_tilt = float(self.p(
            'cable_marker_max_surface_tilt_deg'
            if is_cable_marker else 'max_surface_tilt_deg'))
        if tilt > max_tilt:
            return None, f'surface tilt {tilt:.1f}deg > {max_tilt:.1f}deg'

        limits_min = np.asarray(self.p(
            'cable_marker_workspace_min'
            if is_cable_marker else 'workspace_min'), dtype=float)
        limits_max = np.asarray(self.p(
            'cable_marker_workspace_max'
            if is_cable_marker else 'workspace_max'), dtype=float)
        if np.any(center_b < limits_min) or np.any(center_b > limits_max):
            return None, (
                'position outside perception workspace: '
                f'[{center_b[0]:.3f}, {center_b[1]:.3f}, {center_b[2]:.3f}]')
        if not is_cable_marker and as_bool(self.p('enable_table_filter')):
            if center_b[2] < float(self.p('table_height_m')) + float(self.p('min_object_height_m')):
                return None, 'target rejected by table-height filter'

        polygon = np.asarray(corners).reshape(4, 2)
        edge = polygon[1] - polygon[0]
        kc = np.asarray(self.color_info.k).reshape(3, 3)
        ray_c = np.array([edge[0] / kc[0, 0], edge[1] / kc[1, 1], 0.0])
        edge_b = base_to_camera[:3, :3].dot(ray_c)
        edge_b[2] = 0.0
        if np.linalg.norm(edge_b) < 1e-6:
            edge_b = np.array([1.0, 0.0, 0.0])
        x_axis = edge_b / np.linalg.norm(edge_b)
        yaw_offset = math.radians(float(self.p('grasp_yaw_offset_deg')))
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        x_axis = np.array([c * x_axis[0] - s * x_axis[1], s * x_axis[0] + c * x_axis[1], 0.0])
        z_axis = np.array([0.0, 0.0, -1.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
        x_axis = np.cross(y_axis, z_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return {
            'position': center_b,
            'rotation': rotation,
            'yaw': math.atan2(x_axis[1], x_axis[0]),
        }, ''

    def evaluate_stability(self, history):
        count = int(self.p('stable_frames'))
        if len(history) < count:
            return None
        positions = np.asarray([item[0] for item in history])
        yaws = np.unwrap(np.asarray([item[1] for item in history]))
        if np.max(np.std(positions, axis=0)) > float(self.p('stable_position_std_m')):
            return None
        if math.degrees(np.std(yaws)) > float(self.p('stable_yaw_std_deg')):
            return None
        return {
            'position': positions.mean(axis=0),
            'rotation': history[-1][2],
            'yaw': wrap_angle(float(yaws.mean())),
            'stamp': self.get_clock().now(),
        }

    def tip_matrix(self, target, z_offset):
        matrix = np.eye(4)
        matrix[:3, :3] = target['rotation']
        matrix[:3, 3] = target['position'] + np.array([0.0, 0.0, z_offset])
        return matrix

    def flange_matrix(self, target, z_offset):
        return self.tip_matrix(target, z_offset).dot(np.linalg.inv(self.tool_transform))

    def publish_target(self, marker_id, target):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.base_frame
        pose_msg.pose = pose_from_matrix(self.flange_matrix(target, -float(self.p('grasp_below_surface_m'))))
        self.target_pub.publish(pose_msg)

        markers = MarkerArray()
        for index, (name, offset, color) in enumerate([
            ('surface', 0.0, (0.2, 0.8, 1.0)),
            ('pregrasp', float(self.p('pregrasp_height_m')), (1.0, 0.7, 0.1)),
            ('grasp', -float(self.p('grasp_below_surface_m')), (1.0, 0.1, 0.1)),
        ]):
            marker = Marker()
            marker.header = pose_msg.header
            marker.ns = 'eco65_grasp_' + name
            marker.id = int(marker_id) * 10 + index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose_from_matrix(self.tip_matrix(target, offset))
            marker.scale.x = 0.09
            marker.scale.y = 0.012
            marker.scale.z = 0.012
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            markers.markers.append(marker)
        self.markers_pub.publish(markers)

    def publish_debug(self, source_msg, image):
        if self.debug_pub.get_subscription_count() == 0:
            return
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        msg.header = source_msg.header
        self.debug_pub.publish(msg)

    def fresh_targets(self):
        now = self.get_clock().now()
        timeout = float(self.p('target_timeout_s'))
        with self.lock:
            return {
                key: value for key, value in self.stable_targets.items()
                if (now - value['stamp']).nanoseconds * 1e-9 <= timeout
            }

    def nearest_target(self):
        targets = self.fresh_targets()
        if not targets:
            return None, None
        try:
            current_tip = self.lookup_base_to_flange().dot(self.tool_transform)[:3, 3]
            marker_id = min(targets, key=lambda key: np.linalg.norm(targets[key]['position'] - current_tip))
        except Exception:
            marker_id = sorted(targets)[0]
        return marker_id, targets[marker_id]

    def fresh_target_by_id(self, marker_id):
        targets = self.fresh_targets()
        return targets.get(int(marker_id))

    def clear_targets(self, enable_collection=True):
        with self.lock:
            self.stable_targets.clear()
            self.histories.clear()
            self.target_collection_enabled = enable_collection

    def preview_cb(self, _request, response):
        marker_id, target = self.nearest_target()
        if target is None:
            response.success = False
            response.message = 'No fresh stable target'
            return response
        p = target['position']
        response.success = True
        response.message = f'marker {marker_id} surface in {self.base_frame}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m'
        return response

    def require_motion(self):
        if not self.execute_motion:
            return 'Motion is disabled; relaunch with execute_motion:=true'
        return None

    def call_observation(self):
        error = self.require_motion()
        if error:
            return False, error
        if not self.observation_client.wait_for_service(timeout_sec=3.0):
            return False, f'observation service unavailable: {self.p("observation_service")}'
        future = self.observation_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 20.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, 'observation service timed out'
        result = future.result()
        return bool(result.success), result.message

    def observation_cb(self, _request, response):
        with self.motion_lock:
            response.success, response.message = self.call_observation()
        return response

    def command_gripper(self, closing):
        if serial is None:
            return False, 'python serial module is missing; install python3-serial'
        hex_string = self.p('gripper_close_hex') if closing else self.p('gripper_open_hex')
        try:
            with serial.Serial(
                self.p('gripper_port'),
                int(self.p('gripper_baud')),
                timeout=float(self.p('gripper_timeout_s')),
            ) as stream:
                stream.write(bytes.fromhex(hex_string))
            time.sleep(float(self.p('gripper_motion_wait_s')))
            return True, 'gripper close command sent' if closing else 'gripper open command sent'
        except Exception as exc:
            return False, f'gripper serial command failed: {exc}'

    def open_cb(self, _request, response):
        response.success, response.message = self.command_gripper(False)
        return response

    def close_cb(self, _request, response):
        response.success, response.message = self.command_gripper(True)
        return response

    def latest_joint_map(self):
        with self.lock:
            return dict(self.latest_joints)

    def publish_movej(self, updates: Dict[str, float]):
        joints = self.latest_joint_map()
        if not joints:
            return False, 'No /joint_states received'
        missing = [name for name in self.joint_names if name not in joints]
        if missing:
            return False, 'Joint state missing: ' + ', '.join(missing)
        commanded = {name: float(joints[name]) for name in self.joint_names}
        commanded.update(updates)
        msg = Movej()
        msg.joint = [commanded[name] for name in self.joint_names]
        msg.speed = int(self.p('joint_speed'))
        msg.block = True
        msg.trajectory_connect = 0
        msg.dof = 6
        self.movej_result = None
        self.movej_pub.publish(msg)
        return self.wait_bool_result(lambda: self.movej_result, 'MoveJ')

    def publish_pose_motion(self, matrix, linear=False, speed=None,
                            timeout_s=None):
        msg = Movel() if linear else Movejp()
        msg.pose = pose_from_matrix(matrix)
        msg.speed = int(
            speed if speed is not None else
            self.p('linear_speed') if linear else self.p('pose_speed'))
        msg.block = True
        msg.trajectory_connect = 0
        if linear:
            self.movel_result = None
            self.movel_pub.publish(msg)
            return self.wait_bool_result(
                lambda: self.movel_result, 'MoveL', timeout_s)
        self.movejp_result = None
        self.movejp_pub.publish(msg)
        return self.wait_bool_result(
            lambda: self.movejp_result, 'MoveJ_P', timeout_s)

    def wait_bool_result(self, getter, label, timeout_s=None):
        timeout = (
            float(timeout_s) if timeout_s is not None
            else float(self.p('command_timeout_s')))
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            result = getter()
            if result is not None:
                return bool(result), f'{label} result={result}'
            time.sleep(0.05)
        return False, f'{label} result timed out'

    def call_trigger_client(self, client, service_name, label, timeout_s):
        if not client.wait_for_service(timeout_sec=timeout_s):
            return False, f'{label} service unavailable: {service_name}'
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, f'{label} service timed out'
        result = future.result()
        return bool(result.success), result.message

    def start_compliance_after_lift(self):
        timeout_s = float(self.p('compliance_service_timeout_s'))
        settle_s = max(0.0, float(self.p('compliance_settle_before_zero_s')))
        if settle_s > 0.0:
            self.get_logger().info(
                f'[六维力] 抬升完成，等待 {settle_s:.1f}s 稳定后再 zero_bias')
            time.sleep(settle_s)
        ok, text = self.call_trigger_client(
            self.compliance_zero_client,
            self.p('compliance_zero_service'),
            'sixforce zero_bias',
            timeout_s)
        if not ok:
            return False, text
        ok, text = self.call_trigger_client(
            self.compliance_start_client,
            self.p('compliance_start_service'),
            'sixforce start',
            timeout_s)
        if not ok:
            return False, text
        return True, text

    def wait_for_fresh_target(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            marker_id, target = self.nearest_target()
            if target is not None:
                return marker_id, target
            time.sleep(0.10)
        return None, None

    def wait_for_marker_target(self, marker_id, seconds):
        marker_id = int(marker_id)
        deadline = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            target = self.fresh_target_by_id(marker_id)
            if target is not None:
                return marker_id, target
            time.sleep(0.10)
        return None, None

    def search_for_marker(self, wanted_marker_id):
        wanted_marker_id = int(wanted_marker_id)
        self.clear_targets(True)
        marker_id, target = self.wait_for_marker_target(
            wanted_marker_id,
            float(self.p('base_scan_step_dwell_s')))
        if target is not None:
            self.validated_target = (marker_id, target)
            return marker_id, target, f'stable marker {wanted_marker_id} found at observation pose'
        if not as_bool(self.p('enable_base_scan')):
            return None, None, f'No stable marker {wanted_marker_id} at observation pose'

        joints = self.latest_joint_map()
        scan_joint = self.p('base_scan_joint_name')
        if scan_joint not in joints:
            return None, None, 'Joint state missing scan joint ' + scan_joint
        lower = float(self.p('base_scan_min_rad'))
        upper = float(self.p('base_scan_max_rad'))
        step = abs(float(self.p('base_scan_step_rad')))
        direction = 1.0 if float(self.p('base_scan_initial_direction')) >= 0 else -1.0
        target_joint = min(upper, max(lower, float(joints[scan_joint])))
        deadline = time.monotonic() + float(self.p('base_scan_timeout_s'))
        while rclpy.ok() and time.monotonic() < deadline:
            target_joint += direction * step
            if target_joint >= upper:
                target_joint = upper
                direction = -1.0
            elif target_joint <= lower:
                target_joint = lower
                direction = 1.0
            ok, text = self.publish_movej({scan_joint: target_joint})
            if not ok:
                return None, None, 'base scan failed: ' + text
            time.sleep(float(self.p('base_scan_settle_s')))
            self.clear_targets(True)
            marker_id, target = self.wait_for_marker_target(
                wanted_marker_id,
                float(self.p('base_scan_step_dwell_s')))
            if target is not None:
                self.validated_target = (marker_id, target)
                return marker_id, target, (
                    f'stable marker {wanted_marker_id} found at '
                    f'{scan_joint}={target_joint:.3f}')
        return None, None, f'No stable marker {wanted_marker_id} found before scan timeout'

    def search_for_stable_target(self):
        self.clear_targets(True)
        marker_id, target = self.wait_for_fresh_target(float(self.p('base_scan_step_dwell_s')))
        if target is not None:
            self.validated_target = (marker_id, target)
            return marker_id, target, 'stable target found at observation pose'
        if not as_bool(self.p('enable_base_scan')):
            return None, None, 'No stable target at observation pose'

        joints = self.latest_joint_map()
        scan_joint = self.p('base_scan_joint_name')
        if scan_joint not in joints:
            return None, None, 'Joint state missing scan joint ' + scan_joint
        lower = float(self.p('base_scan_min_rad'))
        upper = float(self.p('base_scan_max_rad'))
        step = abs(float(self.p('base_scan_step_rad')))
        direction = 1.0 if float(self.p('base_scan_initial_direction')) >= 0 else -1.0
        target_joint = min(upper, max(lower, float(joints[scan_joint])))
        deadline = time.monotonic() + float(self.p('base_scan_timeout_s'))
        while rclpy.ok() and time.monotonic() < deadline:
            target_joint += direction * step
            if target_joint >= upper:
                target_joint = upper
                direction = -1.0
            elif target_joint <= lower:
                target_joint = lower
                direction = 1.0
            ok, text = self.publish_movej({scan_joint: target_joint})
            if not ok:
                return None, None, 'base scan failed: ' + text
            time.sleep(float(self.p('base_scan_settle_s')))
            self.clear_targets(True)
            marker_id, target = self.wait_for_fresh_target(float(self.p('base_scan_step_dwell_s')))
            if target is not None:
                self.validated_target = (marker_id, target)
                return marker_id, target, f'stable target found at {scan_joint}={target_joint:.3f}'
        return None, None, 'No stable target found before scan timeout'

    def place_flange_matrix(self, place_target):
        matrix = np.eye(4)
        rotation = np.array(place_target['rotation'], dtype=float)
        yaw_offset = math.radians(float(self.p('place_yaw_offset_deg')))
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        yaw_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        matrix[:3, :3] = yaw_rot.dot(rotation)
        offset_xy = np.asarray(self.p('place_xy_offset_m'), dtype=float)
        matrix[:3, 3] = place_target['position'] + np.array([
            float(offset_xy[0]),
            float(offset_xy[1]),
            float(self.p('place_above_height_m')),
        ])
        return matrix.dot(np.linalg.inv(self.tool_transform))

    def cable_preinsert_tip_matrix(self, marker_target):
        offset = np.asarray(
            self.p('cable_marker_to_preinsert_offset_base_m'), dtype=float)
        if offset.shape != (3,):
            raise ValueError(
                'cable_marker_to_preinsert_offset_base_m must contain 3 values')
        tip_axis = np.asarray(
            self.p('cable_tool_tip_axis_base'), dtype=float)
        x_hint = np.asarray(
            self.p('cable_tool_x_axis_base'), dtype=float)
        if tip_axis.shape != (3,) or x_hint.shape != (3,):
            raise ValueError('cable tool axes must each contain 3 values')
        tip_norm = float(np.linalg.norm(tip_axis))
        if tip_norm < 1e-9:
            raise ValueError('cable_tool_tip_axis_base cannot be zero')
        z_axis = tip_axis / tip_norm
        x_axis = x_hint - z_axis * float(np.dot(x_hint, z_axis))
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-9:
            raise ValueError('cable tool X axis cannot be parallel to tip axis')
        x_axis /= x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)

        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        matrix[:3, 3] = np.asarray(marker_target['position']) + offset
        return matrix

    def publish_cable_preinsert_marker(self, tip_matrix, marker_id):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.base_frame
        marker.ns = 'eco65_cable_preinsert'
        marker.id = int(marker_id)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        # RViz arrow points along local X, so visualize the tool +Z/tip axis.
        arrow_matrix = np.eye(4)
        arrow_matrix[:3, 0] = tip_matrix[:3, 2]
        arrow_matrix[:3, 1] = tip_matrix[:3, 0]
        arrow_matrix[:3, 2] = tip_matrix[:3, 1]
        arrow_matrix[:3, 3] = tip_matrix[:3, 3]
        marker.pose = pose_from_matrix(arrow_matrix)
        marker.scale.x = 0.12
        marker.scale.y = 0.018
        marker.scale.z = 0.018
        marker.color.r = 0.2
        marker.color.g = 1.0
        marker.color.b = 0.3
        marker.color.a = 1.0
        markers = MarkerArray()
        markers.markers.append(marker)
        self.markers_pub.publish(markers)

    def cable_preinsert_target(self):
        marker_id = int(self.p('cable_socket_marker_id'))
        target = self.fresh_target_by_id(marker_id)
        if target is None:
            _, target = self.wait_for_marker_target(
                marker_id, float(self.p('cable_target_wait_s')))
        if target is None:
            return marker_id, None, None, (
                f'No fresh stable cable socket marker {marker_id}')
        try:
            tip_matrix = self.cable_preinsert_tip_matrix(target)
        except (TransformException, ValueError) as exc:
            return marker_id, target, None, f'Cannot compute cable preinsert: {exc}'

        position = tip_matrix[:3, 3]
        workspace_min = np.asarray(self.p('workspace_min'), dtype=float)
        workspace_max = np.asarray(self.p('workspace_max'), dtype=float)
        if (workspace_min.shape != (3,) or workspace_max.shape != (3,) or
                np.any(position < workspace_min) or np.any(position > workspace_max)):
            return marker_id, target, None, (
                'Cable preinsert is outside workspace: '
                f'[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]')
        self.publish_cable_preinsert_marker(tip_matrix, marker_id)
        return marker_id, target, tip_matrix, 'ok'

    def preview_cable_preinsert_cb(self, _request, response):
        marker_id, _target, tip_matrix, error = self.cable_preinsert_target()
        if tip_matrix is None:
            response.success = False
            response.message = error
            return response
        p = tip_matrix[:3, 3]
        response.success = True
        response.message = (
            f'cable marker {marker_id} preinsert TCP in {self.base_frame}: '
            f'[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m; '
            'tool +Z constrained to base Y+')
        return response

    def move_to_cable_preinsert_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            marker_id, _target, tip_matrix, error = self.cable_preinsert_target()
            if tip_matrix is None:
                response.success = False
                response.message = error
                return response
            current_tip = self.lookup_base_to_flange().dot(self.tool_transform)
            distance = float(np.linalg.norm(
                tip_matrix[:3, 3] - current_tip[:3, 3]))
            max_distance = float(self.p('cable_max_preinsert_move_m'))
            if max_distance > 0.0 and distance > max_distance:
                response.success = False
                response.message = (
                    f'Cable preinsert move too far: {distance:.3f}m > '
                    f'{max_distance:.3f}m')
                return response
            flange_matrix = tip_matrix.dot(np.linalg.inv(self.tool_transform))
            self.get_logger().info(
                '[线缆预插接] marker=%d, TCP=(%.4f, %.4f, %.4f), '
                'distance=%.3fm, tool +Z -> base Y+' % (
                    marker_id, *tip_matrix[:3, 3], distance))
            ok, text = self.publish_pose_motion(
                flange_matrix, linear=False,
                speed=int(self.p('cable_preinsert_speed')),
                timeout_s=float(self.p('cable_preinsert_timeout_s')))
            if not ok and text == 'MoveJ_P result timed out':
                try:
                    actual_tip = self.lookup_base_to_flange().dot(
                        self.tool_transform)
                    position_error = float(np.linalg.norm(
                        actual_tip[:3, 3] - tip_matrix[:3, 3]))
                    rotation_delta = actual_tip[:3, :3].T.dot(
                        tip_matrix[:3, :3])
                    cosine = float(np.clip(
                        (np.trace(rotation_delta) - 1.0) * 0.5,
                        -1.0, 1.0))
                    orientation_error_deg = math.degrees(math.acos(cosine))
                    position_tolerance = float(self.p(
                        'cable_preinsert_reached_position_tolerance_m'))
                    orientation_tolerance = float(self.p(
                        'cable_preinsert_reached_orientation_tolerance_deg'))
                    if (position_error <= position_tolerance and
                            orientation_error_deg <= orientation_tolerance):
                        ok = True
                        text = (
                            'MoveJ_P result timed out, but TCP reached target '
                            f'(position error={position_error:.3f}m, '
                            f'orientation error={orientation_error_deg:.1f}deg)')
                    else:
                        text += (
                            f'; TCP not at target (position error='
                            f'{position_error:.3f}m, orientation error='
                            f'{orientation_error_deg:.1f}deg)')
                except TransformException as exc:
                    text += f'; cannot verify current TCP: {exc}'
            if ok:
                self.last_cable_preinsert_tip = np.array(tip_matrix)
            response.success = ok
            response.message = (
                f'moved to cable preinsert for marker {marker_id}: {text}'
                if ok else f'cable preinsert move failed: {text}')
            return response

    def capture_cable_fz_baseline(self):
        window = float(self.p('cable_insert_baseline_window_s'))
        minimum = int(self.p('cable_insert_baseline_min_samples'))
        sample_timeout = float(self.p('force_sample_timeout_s'))
        deadline = time.monotonic() + max(window, sample_timeout, 0.8)
        samples = []
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            samples = [
                value for stamp, value, _sequence in list(self.fz_samples)
                if now - stamp <= window
            ]
            if (len(samples) >= minimum and self.fz_samples and
                    now - self.fz_samples[-1][0] <= sample_timeout):
                baseline = float(np.median(samples))
                self.get_logger().info(
                    f'[线缆力控] Fz baseline={baseline:.3f}N, '
                    f'samples={len(samples)}')
                return baseline, ''
            time.sleep(0.01)
        return None, (
            f'Cannot establish fresh Fz baseline: got {len(samples)} samples, '
            f'need {minimum}; check {self.p("sixforce_topic")}')

    def publish_cable_insert_motion(self, controller_target, baseline):
        # Build the absolute target in the controller's own pose convention.
        # Its native endpoint differs from the URDF Link6/TCP by about 17 mm,
        # so mixing the two conventions can reverse a short insertion move.
        msg = Movel()
        msg.pose = pose_from_matrix(controller_target)
        msg.speed = int(self.p('cable_insert_speed'))
        # A blocking MoveL pauses this controller's UDP force stream. Submit the
        # segment asynchronously and use arm status 1 -> 0 as completion.
        msg.block = False
        msg.trajectory_connect = 0
        self.movel_result = None
        start_sequence = self.fz_sequence
        start_status_sequence = self.arm_status_sequence
        self.movel_pub.publish(msg)

        threshold = float(self.p('cable_insert_force_threshold_n'))
        required = max(1, int(self.p('cable_insert_force_trigger_samples')))
        sample_timeout = float(self.p('force_sample_timeout_s'))
        first_sample_timeout = float(self.p('force_first_sample_timeout_s'))
        deadline = time.monotonic() + float(self.p('command_timeout_s'))
        # The controller can briefly pause UDP reporting while accepting a
        # non-blocking MoveL. Keep checking samples during this transition,
        # but enforce the normal stale timeout only after the grace window.
        startup_grace_deadline = time.monotonic() + first_sample_timeout
        over_count = 0
        last_sequence = start_sequence
        received_post_command_sample = False
        command_accepted = False
        saw_move_l = False
        last_status_sequence = start_status_sequence
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            latest = self.fz_samples[-1] if self.fz_samples else None
            if latest is not None and latest[2] != start_sequence:
                received_post_command_sample = True
            startup_grace_finished = now >= startup_grace_deadline
            if (startup_grace_finished and received_post_command_sample and
                    (latest is None or now - latest[0] > sample_timeout)):
                self.move_stop_pub.publish(Empty())
                age = float('inf') if latest is None else now - latest[0]
                return False, False, (
                    f'Fz stream became stale after command (age={age:.3f}s, '
                    f'last_sequence={last_sequence}, max_callback_gap='
                    f'{self.fz_max_gap_during_insertion:.3f}s); '
                    'move_stop sent')
            if not received_post_command_sample and startup_grace_finished:
                self.move_stop_pub.publish(Empty())
                age = float('inf') if latest is None else now - latest[0]
                return False, False, (
                    'No new Fz sample arrived after MoveL publish '
                    f'(waited {first_sample_timeout:.2f}s, age={age:.3f}s, '
                    f'start_sequence={start_sequence}, '
                    f'current_sequence={self.fz_sequence}); move_stop sent')
            if latest is not None:
                _stamp, value, sequence = latest
            else:
                sequence = last_sequence
                value = baseline
            if received_post_command_sample and sequence != last_sequence:
                last_sequence = sequence
                delta = float(value - baseline)
                over_count = over_count + 1 if abs(delta) >= threshold else 0
                if over_count >= required:
                    self.move_stop_pub.publish(Empty())
                    self.get_logger().warn(
                        f'[线缆力控触发] Fz delta={delta:.3f}N >= '
                        f'{threshold:.3f}N，已发送 move_stop')
                    return True, True, (
                        f'force stop triggered: Fz delta={delta:.3f}N; '
                        'move_stop sent')
            if self.movel_result is not None:
                if not self.movel_result:
                    self.move_stop_pub.publish(Empty())
                    return False, False, (
                        'controller-frame absolute MoveL rejected; move_stop sent')
                command_accepted = True
                self.movel_result = None
            status = self.arm_status
            if self.arm_status_sequence != last_status_sequence:
                last_status_sequence = self.arm_status_sequence
                if status == 1:
                    saw_move_l = True
                elif command_accepted and saw_move_l and status in (9, 10, 11):
                    self.move_stop_pub.publish(Empty())
                    return False, False, (
                        f'MoveL stopped with arm_status={status}; '
                        'move_stop sent')
            force_is_fresh = (
                received_post_command_sample and latest is not None and
                now - latest[0] <= sample_timeout)
            if (command_accepted and saw_move_l and status == 0 and
                    force_is_fresh):
                return True, False, (
                    'Controller-frame insertion segment completed: arm_status 1->0; '
                    f'Fz threshold {threshold:.1f}N not reached')
            time.sleep(0.002)

        self.move_stop_pub.publish(Empty())
        return False, False, (
            'Cable insertion timed out; move_stop sent; '
            f'accepted={command_accepted}, saw_move_l={saw_move_l}, '
            f'arm_status={self.arm_status}')

    @contextmanager
    def cable_insertion_guard(self):
        self.fz_max_gap_during_insertion = 0.0
        self.cable_insertion_active.set()
        try:
            yield
        finally:
            self.cable_insertion_active.clear()

    def insert_cable_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock, self.cable_insertion_guard():
            if self.arm_status != 0:
                response.success = False
                response.message = (
                    f'Robot is not idle: arm_status={self.arm_status}; '
                    'clear stop/alarm and move to cable preinsert again')
                return response
            if self.last_cable_preinsert_tip is None:
                response.success = False
                response.message = (
                    'No successful cable preinsert recorded; call '
                    '/eco65_visual_grasp/move_to_cable_preinsert first')
                return response
            try:
                current_tip = self.lookup_base_to_flange().dot(self.tool_transform)
            except TransformException as exc:
                response.success = False
                response.message = f'Cannot read current TCP: {exc}'
                return response
            start_error = float(np.linalg.norm(
                current_tip[:3, 3] - self.last_cable_preinsert_tip[:3, 3]))
            tolerance = float(self.p('cable_insert_start_tolerance_m'))
            if start_error > tolerance:
                response.success = False
                response.message = (
                    f'TCP is {start_error:.3f}m from recorded preinsert '
                    f'(limit {tolerance:.3f}m)')
                return response
            insertion_start_tip = np.array(current_tip)

            with self.lock:
                controller_start = (
                    None if self.controller_pose_matrix is None else
                    np.array(self.controller_pose_matrix))
                controller_pose_stamp = self.controller_pose_stamp
            controller_pose_age = (
                float('inf') if controller_pose_stamp is None else
                time.monotonic() - controller_pose_stamp)
            controller_pose_timeout = float(self.p('controller_pose_timeout_s'))
            if (controller_start is None or
                    controller_pose_age > controller_pose_timeout):
                response.success = False
                response.message = (
                    f'Controller pose is missing/stale (age={controller_pose_age:.3f}s); '
                    f'check {self.p("controller_pose_topic")}')
                return response

            baseline, error = self.capture_cable_fz_baseline()
            if baseline is None:
                response.success = False
                response.message = error
                return response
            if as_bool(self.p('cable_insert_use_tool_tip_axis')):
                axis = np.asarray(
                    controller_start[:3, 2], dtype=float)
                axis *= float(self.p('cable_insert_tool_axis_sign'))
                axis_source = 'controller native tool +Z axis'
            else:
                axis = np.asarray(
                    self.p('cable_insert_axis_base'), dtype=float)
                axis_source = 'fixed base axis'
            axis_norm = float(np.linalg.norm(axis))
            if axis.shape != (3,) or axis_norm < 1e-9:
                response.success = False
                response.message = 'Invalid cable_insert_axis_base'
                return response
            axis /= axis_norm
            expected_axis = np.asarray(
                self.p('cable_insert_axis_base'), dtype=float)
            expected_norm = float(np.linalg.norm(expected_axis))
            if expected_axis.shape != (3,) or expected_norm < 1e-9:
                response.success = False
                response.message = 'Invalid cable_insert_axis_base'
                return response
            expected_axis /= expected_norm
            axis_alignment = float(np.dot(axis, expected_axis))
            min_alignment = float(self.p('cable_insert_axis_min_alignment'))
            if axis_alignment < min_alignment:
                response.success = False
                response.message = (
                    'Insertion direction safety check failed: controller tool '
                    f'axis={axis.round(3).tolist()}, expected base axis='
                    f'{expected_axis.round(3).tolist()}, alignment='
                    f'{axis_alignment:.3f} < {min_alignment:.3f}; no motion sent')
                return response
            distance = float(self.p('cable_insert_distance_m'))
            step = float(self.p('cable_insert_step_m'))
            if distance <= 0.0 or step <= 0.0:
                response.success = False
                response.message = (
                    'cable_insert_distance_m and cable_insert_step_m must be positive')
                return response
            segment_count = int(math.ceil(distance / step))
            self.get_logger().info(
                '[线缆插接] 沿 base (%.3f, %.3f, %.3f) 最多移动 %.3fm，'
                '分%d段，每段最大%.3fm，Fz相对阈值=%.1fN，'
                '起点=(%.4f, %.4f, %.4f)，方向来源=%s' % (
                    *axis, distance, segment_count, step,
                    float(self.p('cable_insert_force_threshold_n')),
                    *insertion_start_tip[:3, 3], axis_source))
            self.get_logger().info(
                f'[线缆插接] 方向安全检查通过，alignment={axis_alignment:.3f}')
            self.get_logger().info(
                '[线缆插接] 控制器原生起点=(%.4f, %.4f, %.4f)，'
                'TF安全监测起点=(%.4f, %.4f, %.4f)' % (
                    *controller_start[:3, 3], *insertion_start_tip[:3, 3]))
            moved = 0.0
            for index in range(segment_count):
                try:
                    current_tip = self.lookup_base_to_flange().dot(
                        self.tool_transform)
                except TransformException as exc:
                    self.move_stop_pub.publish(Empty())
                    response.success = False
                    response.message = (
                        f'Cannot read TCP before segment {index + 1}: {exc}; '
                        'move_stop sent')
                    return response
                max_overshoot = float(self.p('cable_insert_max_overshoot_m'))
                measured_from_start = current_tip[:3, 3] - insertion_start_tip[:3, 3]
                measured_distance = float(np.linalg.norm(measured_from_start))
                if measured_distance > distance + max_overshoot:
                    self.move_stop_pub.publish(Empty())
                    response.success = False
                    response.message = (
                        f'TCP exceeded insertion limit before segment '
                        f'{index + 1}: {measured_distance:.3f}m > '
                        f'{distance + max_overshoot:.3f}m; move_stop sent')
                    return response
                segment = min(step, distance - moved)
                planned_distance = moved + segment
                # Every waypoint is absolute relative to the single insertion
                # start pose. This prevents TF lag from accumulating distance.
                target_tip = np.array(insertion_start_tip)
                target_tip[:3, 3] += axis * planned_distance
                controller_target = np.array(controller_start)
                controller_target[:3, 3] += axis * planned_distance
                self.get_logger().info(
                    '[线缆插接] 下发 %d/%d 段控制器目标=(%.4f, %.4f, %.4f)，'
                    'TF安全目标=(%.4f, %.4f, %.4f)，计划累计=%.3fm' % (
                        index + 1, segment_count,
                        *controller_target[:3, 3], *target_tip[:3, 3],
                        planned_distance))
                try:
                    ok, force_triggered, text = self.publish_cable_insert_motion(
                        controller_target, baseline)
                except Exception as exc:
                    self.move_stop_pub.publish(Empty())
                    self.get_logger().exception(
                        '[线缆插接] 构造或发送 MoveL 时发生异常')
                    response.success = False
                    response.message = (
                        f'Insertion segment {index + 1}/{segment_count} '
                        f'command exception: {type(exc).__name__}: {exc}; '
                        'move_stop sent')
                    return response
                if force_triggered:
                    response.success = True
                    response.message = (
                        f'{text}; inserted approximately {moved:.3f}m before contact')
                    return response
                if not ok:
                    response.success = False
                    response.message = (
                        f'Insertion segment {index + 1}/{segment_count} failed '
                        f'after approximately {moved:.3f}m: {text}')
                    return response
                verify_deadline = time.monotonic() + 1.0
                actual_tip = None
                target_error = float('inf')
                while rclpy.ok() and time.monotonic() < verify_deadline:
                    try:
                        candidate = self.lookup_base_to_flange().dot(
                            self.tool_transform)
                    except TransformException:
                        time.sleep(0.02)
                        continue
                    candidate_error = float(np.linalg.norm(
                        candidate[:3, 3] - target_tip[:3, 3]))
                    actual_tip = candidate
                    target_error = candidate_error
                    if candidate_error <= float(
                            self.p('cable_insert_position_tolerance_m')):
                        break
                    time.sleep(0.02)
                if actual_tip is None:
                    self.move_stop_pub.publish(Empty())
                    response.success = False
                    response.message = (
                        f'Cannot verify TCP after insertion segment '
                        f'{index + 1}; move_stop sent')
                    return response
                actual_delta = actual_tip[:3, 3] - insertion_start_tip[:3, 3]
                actual_distance = float(np.linalg.norm(actual_delta))
                axial_distance = float(np.dot(actual_delta, axis))
                lateral_distance = float(np.linalg.norm(
                    actual_delta - axial_distance * axis))
                self.get_logger().info(
                    '[线缆插接] 实测 %d/%d TCP=(%.4f, %.4f, %.4f)，'
                    '轴向=%.3fm，总位移=%.3fm，侧向=%.3fm，目标误差=%.3fm' % (
                        index + 1, segment_count, *actual_tip[:3, 3],
                        axial_distance, actual_distance, lateral_distance,
                        target_error))
                if actual_distance > distance + max_overshoot:
                    self.move_stop_pub.publish(Empty())
                    response.success = False
                    response.message = (
                        f'TCP exceeded insertion limit after segment '
                        f'{index + 1}: {actual_distance:.3f}m > '
                        f'{distance + max_overshoot:.3f}m; move_stop sent')
                    return response
                position_tolerance = float(
                    self.p('cable_insert_position_tolerance_m'))
                if target_error > position_tolerance:
                    self.move_stop_pub.publish(Empty())
                    response.success = False
                    response.message = (
                        f'Insertion segment {index + 1} ended '
                        f'{target_error:.3f}m from its absolute target '
                        f'(limit {position_tolerance:.3f}m); move_stop sent')
                    return response
                moved = planned_distance
                self.get_logger().info(
                    f'[线缆插接] 完成 {index + 1}/{segment_count} 段，'
                    f'累计约 {moved:.3f}m')
            response.success = True
            response.message = (
                f'Cable insertion completed {moved:.3f}m without reaching '
                f'{float(self.p("cable_insert_force_threshold_n")):.1f}N')
            return response

    def search_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            ok, text = self.call_observation()
            if not ok:
                response.success = False
                response.message = 'Observation failed: ' + text
                return response
            self.command_gripper(False)
            marker_id, target, text = self.search_for_stable_target()
            response.success = target is not None
            response.message = text if target is None else f'{text}; marker {marker_id} ready'
            return response

    def move_pregrasp_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            marker_id, target = self.nearest_target()
            if target is None and self.validated_target is not None:
                marker_id, target = self.validated_target
            if target is None:
                response.success = False
                response.message = 'No fresh stable target'
                return response
            pregrasp = self.flange_matrix(target, float(self.p('pregrasp_height_m')))
            response.success, response.message = self.publish_pose_motion(pregrasp, linear=False)
            if response.success:
                self.validated_target = (marker_id, target)
                response.message = f'moved to pregrasp for marker {marker_id}: ' + response.message
            return response

    def complete_from_pregrasp_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            if self.validated_target is None:
                marker_id, target = self.nearest_target()
            else:
                marker_id, target = self.validated_target
            if target is None:
                response.success = False
                response.message = 'No target cached from pregrasp/search'
                return response
            grasp = self.flange_matrix(target, -float(self.p('grasp_below_surface_m')))
            lift = self.flange_matrix(target, float(self.p('lift_height_m')))
            ok, text = self.command_gripper(False)
            if not ok:
                response.success = False
                response.message = 'Open failed: ' + text
                return response
            ok, text = self.publish_pose_motion(grasp, linear=True)
            if not ok:
                response.success = False
                response.message = 'Cartesian descent failed: ' + text
                return response
            ok, text = self.command_gripper(True)
            if not ok:
                response.success = False
                response.message = 'Close failed: ' + text
                return response
            ok, text = self.publish_pose_motion(lift, linear=True)
            if not ok:
                response.success = False
                response.message = 'Lift failed: ' + text
                return response
            if as_bool(self.p('start_compliance_after_lift')):
                ok, text = self.start_compliance_after_lift()
                if not ok:
                    response.success = False
                    response.message = 'Sixforce compliance start failed after lift: ' + text
                    return response
            if as_bool(self.p('release_after_lift')):
                time.sleep(float(self.p('release_wait_s')))
                ok, text = self.command_gripper(False)
                if not ok:
                    response.success = False
                    response.message = 'Release failed: ' + text
                    return response
            response.success = True
            response.message = f'completed grasp sequence from pregrasp for marker {marker_id}'
            return response

    def grasp_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            ok, text = self.call_observation()
            if not ok:
                response.success = False
                response.message = 'Observation failed: ' + text
                return response
            self.command_gripper(False)
            marker_id, target, text = self.search_for_stable_target()
            if target is None:
                response.success = False
                response.message = text
                return response
            pregrasp = self.flange_matrix(target, float(self.p('pregrasp_height_m')))
            ok, text = self.publish_pose_motion(pregrasp, linear=False)
            if not ok:
                response.success = False
                response.message = 'Pregrasp failed: ' + text
                return response
            self.validated_target = (marker_id, target)
        return self.complete_from_pregrasp_cb(_request, response)

    def grasp_object_and_move_to_place_cb(self, _request, response):
        error = self.require_motion()
        if error:
            response.success = False
            response.message = error
            return response
        with self.motion_lock:
            object_marker_id = int(self.p('object_marker_id'))
            place_marker_id = int(self.p('place_marker_id'))
            ok, text = self.call_observation()
            if not ok:
                response.success = False
                response.message = 'Observation failed: ' + text
                return response
            self.command_gripper(False)

            self.clear_targets(True)
            _, place_target = self.wait_for_marker_target(
                place_marker_id,
                float(self.p('place_target_wait_s')))
            if place_target is None:
                response.success = False
                response.message = f'No fresh stable place marker {place_marker_id}'
                return response
            p = place_target['position']
            self.get_logger().info(
                f'[放置点] 已记录 marker {place_marker_id} in {self.base_frame}: '
                f'[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]')

            marker_id, object_target, text = self.search_for_marker(object_marker_id)
            if object_target is None:
                response.success = False
                response.message = text
                return response

            pregrasp = self.flange_matrix(object_target, float(self.p('pregrasp_height_m')))
            ok, text = self.publish_pose_motion(pregrasp, linear=False)
            if not ok:
                response.success = False
                response.message = 'Pregrasp failed: ' + text
                return response

            grasp = self.flange_matrix(object_target, -float(self.p('grasp_below_surface_m')))
            lift = self.flange_matrix(object_target, float(self.p('lift_height_m')))
            ok, text = self.publish_pose_motion(grasp, linear=True)
            if not ok:
                response.success = False
                response.message = 'Cartesian descent failed: ' + text
                return response
            ok, text = self.command_gripper(True)
            if not ok:
                response.success = False
                response.message = 'Close failed: ' + text
                return response
            ok, text = self.publish_pose_motion(lift, linear=True)
            if not ok:
                response.success = False
                response.message = 'Lift failed: ' + text
                return response

            place_above = self.place_flange_matrix(place_target)
            ok, text = self.publish_pose_motion(place_above, linear=False)
            if not ok:
                response.success = False
                response.message = 'Move to place marker failed: ' + text
                return response
            response.success = True
            response.message = (
                f'grasped marker {marker_id} and moved above place marker '
                f'{place_marker_id} by {float(self.p("place_above_height_m")):.3f} m')
            return response


def main(args=None):
    rclpy.init(args=args)
    node = Eco65VisualGrasp()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
