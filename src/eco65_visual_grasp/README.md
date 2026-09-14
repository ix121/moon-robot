# ECO65 视觉抓取使用说明

本包用于瑞尔曼 ECO65 机械臂的眼在手上视觉抓取，当前配置为：

- 机械臂：Realman ECO65
- 相机：Orbbec Gemini 336
- 目标：ArUco，默认字典 `DICT_4X4_50`
- 夹爪：自研 USB 串口夹爪，设备口 `/dev/ttyACM0`
- 手眼文件：`/home/proton/eco65_grasp_ws/handeye_config.json`
- 协同搬运：抓取抬升后自动启动六维力柔顺，推荐配合 `sixforce_compliance_5d_no_fz.yaml`

抓取流程参考了 Piper 视觉抓取和 RM65-B ROS1 视觉抓取程序：

```text
到观察位
-> 打开夹爪
-> 检测 ArUco
-> 深度拟合目标表面
-> 手眼转换到机械臂坐标系
-> 生成预抓取位和抓取位
-> 到预抓取位
-> 直线下探
-> 闭合夹爪
-> 抬升
-> 启动六维力柔顺
-> 保持夹爪闭合，进入协同搬运
```

## 1. 工作空间

当前包路径：

```text
/home/proton/eco65_grasp_ws/src/eco65_visual_grasp
```

主要文件：

```text
config/eco65_visual_grasp.yaml
launch/eco65_visual_grasp.launch.py
eco65_visual_grasp/eco65_visual_grasp_node.py
```

## 2. 构建

```bash
cd /home/proton/eco65_grasp_ws
source /opt/ros/humble/setup.bash
source /home/proton/rm_arm_ws/install/setup.bash
source /home/proton/cam_ws/install/setup.bash
colcon build --symlink-install
```

修改 Python 或 YAML 后，建议重新执行一次上面的构建命令。

特别注意：launch 默认读取的是 `install` 目录中的配置文件，而不是 `src` 目录中的配置文件。比如修改了：

```text
/home/proton/eco65_grasp_ws/src/eco65_visual_grasp/config/eco65_visual_grasp.yaml
```

必须重新编译：

```bash
cd /home/proton/eco65_grasp_ws
source /opt/ros/humble/setup.bash
source /home/proton/rm_arm_ws/install/setup.bash
source /home/proton/cam_ws/install/setup.bash
colcon build --packages-select eco65_visual_grasp
source install/setup.bash
```

否则运行时仍会使用旧的：

```text
/home/proton/eco65_grasp_ws/install/eco65_visual_grasp/share/eco65_visual_grasp/config/eco65_visual_grasp.yaml
```

## 3. 关键配置

配置文件：

```text
/home/proton/eco65_grasp_ws/src/eco65_visual_grasp/config/eco65_visual_grasp.yaml
```

当前重要参数：

```yaml
base_frame: "baselink"
flange_frame: "Link6"
handeye_file: "/home/proton/eco65_grasp_ws/handeye_config.json"
tool_xyz: [0.0, 0.0, 0.155]
grasp_yaw_offset_deg: 0.0
depth_registered_to_color: false
execute_motion: false
release_after_lift: false
start_compliance_after_lift: true
compliance_settle_before_zero_s: 1.5
object_marker_id: 6
place_marker_id: 0
place_above_height_m: 0.040
```

其中：

- `tool_xyz` 是夹爪末端相对机械臂法兰/`Link6` 的偏移，当前按 `15.5 cm`。
- `grasp_yaw_offset_deg` 是夹爪安装方向补偿，当前为 `0`。
- `depth_registered_to_color: false` 表示使用 Gemini 原始深度图，需要 `/camera/depth_to_color` 外参。
- `execute_motion` 在 YAML 中保持 `false`，实际是否运动通过 launch 参数控制。
- `release_after_lift: false` 表示抬升后不自动松爪，适合搬运。
- `start_compliance_after_lift: true` 表示抓取并抬升后自动调用六维力柔顺节点的 `zero_bias` 和 `start`。
- `compliance_settle_before_zero_s` 表示抬升完成后等待物体晃动稳定再执行 `zero_bias`，当前为 `1.5s`。
- `object_marker_id` 是要抓取的物体 ArUco ID，当前默认 `6`。
- `place_marker_id` 是放置位置 ArUco ID，当前默认 `0`。
- `place_above_height_m` 是搬运到放置码上方的高度，当前先设为 `0.040 m`。

如果现场发现抓取方向不对，可以优先调：

```yaml
grasp_yaw_offset_deg
```

如果下探位置前后有偏差，可以优先调：

```yaml
tool_xyz
pregrasp_height_m
grasp_below_surface_m
```

## 4. 启动顺序

### 终端 1：启动 ECO65 驱动、控制和 MoveIt

```bash
cd /home/proton/rm_grasp_ws
source /opt/ros/humble/setup.bash
source /home/proton/rm_arm_ws/install/setup.bash
source install/setup.bash
ros2 launch rm_eco65_grasp eco65_grasp_bringup.launch.py
```

该 launch 会启动 ECO65 驱动、`rm_control`、MoveIt 和之前写好的观察位服务。

### 终端 2：启动 Gemini 336 相机

```bash
source /opt/ros/humble/setup.bash
source /home/proton/cam_ws/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

检查相机 topic：

```bash
ros2 topic list | grep camera
```

应至少能看到：

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/camera_info
/camera/depth_to_color
```

### 终端 3：启动视觉抓取节点

第一次建议只开感知，不让机械臂运动：

```bash
cd /home/proton/eco65_grasp_ws
source /opt/ros/humble/setup.bash
source /home/proton/rm_arm_ws/install/setup.bash
source /home/proton/cam_ws/install/setup.bash
source install/setup.bash
ros2 launch eco65_visual_grasp eco65_visual_grasp.launch.py execute_motion:=false
```

确认视觉正常后，再重启为允许运动：

```bash
ros2 launch eco65_visual_grasp eco65_visual_grasp.launch.py execute_motion:=true
```

### 终端 4：启动六维力柔顺节点

协同搬运时，需要在执行抓取前先启动六维力节点。这里不要手动调用 `/sixforce_compliance/start`，视觉抓取节点会在夹爪闭合并抬升后自动调用。

推荐使用关闭 Fz 的 5D 配置：

```bash
cd /home/proton/sixforce_ws
source /opt/ros/humble/setup.bash
source /home/proton/rm_arm_ws/install/setup.bash
source install/setup.bash

ros2 launch rm_sixforce_compliance sixforce_compliance.launch.py \
  enable_motion:=true \
  params_file:=/home/proton/sixforce_ws/install/rm_sixforce_compliance/share/rm_sixforce_compliance/config/sixforce_compliance_5d_no_fz.yaml
```

这个配置在基坐标系下打开 `Fx/Fy/Mx/My/Mz`，关闭 `Fz`，避免物体自重被当成竖直方向外力导致末端持续下沉。

## 5. 视觉状态检查

查看视觉节点状态：

```bash
ros2 topic echo /eco65_visual_grasp/status
```

常见状态：

```text
waiting for camera images/info
waiting for /camera/depth_to_color
no ArUco marker detected
marker 6 detected; waiting for stability
stable marker 6 ready
marker 6 rejected by depth/workspace filters
```

看到下面这类输出，说明视觉链路已经正常：

```text
stable marker 6 ready
```

查看当前目标：

```bash
ros2 service call /eco65_visual_grasp/preview std_srvs/srv/Trigger "{}"
ros2 topic echo /eco65_visual_grasp/target_pose --once
```

查看调试图：

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view
```

## 6. 夹爪测试

视觉抓取节点封装了夹爪开合服务。

打开夹爪：

```bash
ros2 service call /eco65_visual_grasp/open_gripper std_srvs/srv/Trigger "{}"
```

闭合夹爪：

```bash
ros2 service call /eco65_visual_grasp/close_gripper std_srvs/srv/Trigger "{}"
```

底层串口命令为：

```bash
# 闭合
python3 -c 'import serial; s=serial.Serial("/dev/ttyACM0",115200,timeout=1); s.write(bytes.fromhex("7b01020120492000c8f87d")); s.close()'

# 张开
python3 -c 'import serial; s=serial.Serial("/dev/ttyACM0",115200,timeout=1); s.write(bytes.fromhex("7b01020020492000c8f97d")); s.close()'
```

如果串口权限不足：

```bash
sudo chmod 666 /dev/ttyACM0
```

长期方案：

```bash
sudo usermod -aG dialout $USER
```

执行后需要重新登录系统。

## 7. 分步抓取测试

第一次上机建议不要直接完整抓取，按下面顺序逐步验证。

### 7.1 到观察位

```bash
ros2 service call /eco65_visual_grasp/move_to_observation std_srvs/srv/Trigger "{}"
```

### 7.2 打开夹爪

```bash
ros2 service call /eco65_visual_grasp/open_gripper std_srvs/srv/Trigger "{}"
```

### 7.3 搜索目标

```bash
ros2 service call /eco65_visual_grasp/search_target std_srvs/srv/Trigger "{}"
```

该服务会：

```text
回观察位
-> 打开夹爪
-> 等待稳定 ArUco
-> 如果观察位看不到，则小幅转动 joint1 搜索
```

### 7.4 移动到预抓取位

```bash
ros2 service call /eco65_visual_grasp/move_to_pregrasp std_srvs/srv/Trigger "{}"
```

到预抓取位后，人工确认：

- 夹爪方向是否正确；
- 夹爪是否对准目标；
- 下探路径下方是否没有障碍物；
- 相机线和夹爪线不会被拉扯。

### 7.5 从预抓取位完成抓取

```bash
ros2 service call /eco65_visual_grasp/complete_from_pregrasp std_srvs/srv/Trigger "{}"
```

该服务会：

```text
打开夹爪
-> 直线下探
-> 闭合夹爪
-> 直线抬升
-> 自动执行 sixforce zero_bias
-> 自动执行 sixforce start
-> 保持夹爪闭合
```

## 8. 一次完整抓取

分步流程确认没问题后，可以一条命令执行完整抓取：

```bash
ros2 service call /eco65_visual_grasp/grasp_nearest std_srvs/srv/Trigger "{}"
```

完整流程为：

```text
到观察位
-> 打开夹爪
-> 搜索稳定目标
-> 到预抓取位
-> 直线下探
-> 闭合夹爪
-> 抬升
-> 自动执行 sixforce zero_bias
-> 自动执行 sixforce start
-> 保持夹爪闭合，进入协同搬运
```

如果暂时只想做普通抓取并在抬升后松爪，把配置改回：

```yaml
start_compliance_after_lift: false
release_after_lift: true
```

## 9. 前期调试：抓 ID6 并移动到 ID0 上方

如果只做单臂前期调试，可以使用：

```bash
ros2 service call /eco65_visual_grasp/grasp_object_and_move_to_place std_srvs/srv/Trigger "{}"
```

该服务会：

```text
到观察位
-> 识别并记录 place_marker_id，也就是默认 ID0，在 baselink 下的位置
-> 识别 object_marker_id，也就是默认 ID6
-> 抓取 ID6
-> 抬升
-> 主动运动到 ID0 上方 place_above_height_m
-> 保持夹爪闭合
```

当前默认参数：

```yaml
object_marker_id: 6
place_marker_id: 0
place_above_height_m: 0.040
place_xy_offset_m: [0.0, 0.0]
place_yaw_offset_deg: 0.0
```

比赛现场如果更换码号或放置高度，只需要改 YAML 里的这些参数并重新编译。

## 10. 常见问题

### 10.1 一直显示 `waiting for /camera/depth_to_color`

检查相机外参 topic：

```bash
ros2 topic list | grep depth_to_color
ros2 topic info /camera/depth_to_color -v
```

正常类型应为：

```text
orbbec_camera_msgs/msg/Extrinsics
```

启动视觉节点的终端必须 source：

```bash
source /home/proton/cam_ws/install/setup.bash
```

### 10.2 `No fresh stable target`

说明当前没有稳定可抓取目标。检查：

```bash
ros2 topic echo /eco65_visual_grasp/status
rqt_image_view /eco65_visual_grasp/debug_image
```

可能原因：

- ArUco 不在视野中；
- ArUco 字典不是 `DICT_4X4_50`；
- 光照反光严重；
- 深度图在 ArUco 区域无效；
- 目标被工作空间过滤；
- 手眼或 TF 坐标不对。

### 10.3 视觉正常但机械臂不动

确认视觉节点是运动模式：

```bash
ros2 param get /eco65_visual_grasp execute_motion
```

必须为：

```text
Boolean value is: True
```

同时确认 ECO65 bringup 已启动，且 `/joint_states` 有输出：

```bash
ros2 topic echo /joint_states --once
```

### 10.4 夹爪没反应

检查串口：

```bash
ls /dev/ttyACM*
```

测试原始命令：

```bash
python3 -c 'import serial; s=serial.Serial("/dev/ttyACM0",115200,timeout=1); s.write(bytes.fromhex("7b01020020492000c8f97d")); s.close()'
```

如果提示权限不足：

```bash
sudo chmod 666 /dev/ttyACM0
```

### 10.5 抓取位置偏高或偏低

优先调整：

```yaml
tool_xyz
pregrasp_height_m
grasp_below_surface_m
lift_height_m
```

当前夹爪末端长度配置为：

```yaml
tool_xyz: [0.0, 0.0, 0.155]
```

### 10.6 夹爪方向不对

调整：

```yaml
grasp_yaw_offset_deg
```

当前为：

```yaml
grasp_yaw_offset_deg: 0.0
```

如果需要顺时针旋转 90 度，通常改为：

```yaml
grasp_yaw_offset_deg: -90.0
```

## 11. 安全注意

每次执行真实运动前确认：

- 急停可用；
- 机械臂运动范围内无人；
- 夹爪和相机线缆不会被拉扯；
- `/joint_states` 正常发布；
- 相机图像和深度图正常；
- `/eco65_visual_grasp/status` 已显示稳定目标；
- 第一次新目标抓取前先执行分步流程，不要直接完整抓取。
