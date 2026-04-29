import isaacgym
from isaacgym import gymapi
import math
import time
import numpy as np  # 💥 新增
import cv2
import os
import socket
def main():
    
## ==========================================
    # 📡 0. 初始化 TCP 神经基站 (等待躯干连接)
    # ==========================================
    TARGET_PORT = 8888
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", TARGET_PORT))
    server_sock.listen(1)
    server_sock.setblocking(False) # 非阻塞模式，不卡顿渲染
    print(f"📡 TCP 大脑已就绪，正在监听端口 {TARGET_PORT}，等待躯干接入...")
    
    client_conn = None
    last_send_time = time.time()
    send_rate = 0.02#，防止塞爆 CAN 总线

    print("🚀 启动：Piper 双臂全自动数据采集测试场")
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # ==========================================
    # 🛡️ 物理引擎终极加固 (告别穿模与软塌)
    # ==========================================
    sim_params.physx.use_gpu = True 
    sim_params.physx.num_position_iterations = 16 
    sim_params.physx.num_velocity_iterations = 4   
    sim_params.physx.contact_offset = 0.002        # 缩小结界，让物体紧密贴合
    sim_params.physx.max_depenetration_velocity = 5.0 
    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ==========================================
    # 🎨 终极材质与物理实体生成配置
    # ==========================================
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.flip_visual_attachments = True 
    asset_options.use_mesh_materials = True
    asset_options.collapse_fixed_joints = True
    
    # 强制凸分解，防穿模
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 300000 
    
    # 动力学稳固三件套 (防脱臼、防鬼畜)
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.thickness = 0.001 
    asset_options.use_physx_armature = True
    asset_options.armature = 0.01 

    asset_root = "./piper_new_assets/asset"
    robot_urdf_file = "urdf/piper_x_description_isaacgym.urdf"
    robot_asset = gym.load_asset(sim, asset_root, robot_urdf_file, asset_options)
    num_dofs = gym.get_asset_dof_count(robot_asset)

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)

    # ==========================================
    # 📦 生成抓取目标并涂上“防滑粉”
    # ==========================================
    box_opts = gymapi.AssetOptions()
    box_opts.density = 400.0 
    box_asset = gym.create_box(sim, 0.04, 0.04, 0.04, box_opts)
    
    # 增强木块摩擦力
    box_shape_props = gym.get_asset_rigid_shape_properties(box_asset)
    for p in box_shape_props:
        p.friction = 2.0  
        p.rolling_friction = 0.1
        p.torsion_friction = 0.1
    gym.set_asset_rigid_shape_properties(box_asset, box_shape_props)
    
    # 左侧木块（红色）
    box_left_pose = gymapi.Transform()
    box_left_pose.p = gymapi.Vec3(0.4, 0.2, 0.02) 
    box_left_pose.r = gymapi.Quat(0, 0, 0, 1)
    box_left_handle = gym.create_actor(env, box_asset, box_left_pose, "box_left", 0, 0)
    gym.set_rigid_body_color(env, box_left_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.2))

    # 右侧木块（蓝色）
    box_right_pose = gymapi.Transform()
    box_right_pose.p = gymapi.Vec3(0.4, -0.2, 0.02)
    box_right_pose.r = gymapi.Quat(0, 0, 0, 1)
    box_right_handle = gym.create_actor(env, box_asset, box_right_pose, "box_right", 0, 0)
    gym.set_rigid_body_color(env, box_right_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.0, 0.8))

    # ==========================================
    # 🤖 部署双臂 & 注入终极物理参数
    # ==========================================
    axis = gymapi.Vec3(0.0, 0.0, 1.0) 
    angle = -math.pi / 2               
    base_rotation = gymapi.Quat.from_axis_angle(axis, angle)

    left_pose = gymapi.Transform()
    left_pose.p = gymapi.Vec3(0.0, 0.3, 0.05) 
    left_pose.r = base_rotation
    left_handle = gym.create_actor(env, robot_asset, left_pose, "left_piper", 0, 1)

    right_pose = gymapi.Transform()
    right_pose.p = gymapi.Vec3(0.0, -0.3, 0.05) 
    right_pose.r = base_rotation
    right_handle = gym.create_actor(env, robot_asset, right_pose, "right_piper", 0, 1)

    # 增强机械臂摩擦力
    robot_shape_props = gym.get_asset_rigid_shape_properties(robot_asset)
    for p in robot_shape_props:
        p.friction = 2.0  
        p.rolling_friction = 0.1
        p.torsion_friction = 0.1
    gym.set_asset_rigid_shape_properties(robot_asset, robot_shape_props)

    for handle in [left_handle, right_handle]:
        # --- 1. 强行增重微小零件 ---
        body_props = gym.get_actor_rigid_body_properties(env, handle)
        for b in range(len(body_props)):
            if body_props[b].mass < 0.1:
                body_props[b].mass = 0.1
        gym.set_actor_rigid_body_properties(env, handle, body_props)

        # --- 2. 注入力量与分离刚度 ---
        props = gym.get_actor_dof_properties(env, handle)
        props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
        props["effort"].fill(1000.0) # 撕毁封印，注入 1000 牛米力量
        
        for j in range(6): # 大臂
            props["stiffness"][j] = 2500.0
            props["damping"][j] = 200.0
        for j in range(6, num_dofs): # 夹爪
            props["stiffness"][j] = 800.0  
            props["damping"][j] = 80.0     
            
        gym.set_actor_dof_properties(env, handle, props)
# ==========================================
    # 📷 部署 VLA 数据采集专用虚拟相机
    # ==========================================
    camera_props = gymapi.CameraProperties()
    camera_props.width = 640   # 常见的 VLA 训练分辨率 (比如 RT-1 用的是 320x256, 这里可以先看高清点)
    camera_props.height = 480
    camera_handle = gym.create_camera_sensor(env, camera_props)
    
    # 设定相机的绝对坐标 (类似在操作台正前方架设的三脚架)
    camera_pos = gymapi.Vec3(1.0, 0.0, 0.6)      # 在机器人正前方 1米，高 0.6米
    camera_target = gymapi.Vec3(0.0, 0.0, 0.05)  # 镜头死死盯住操作台中心(木块区域)
    gym.set_camera_location(camera_handle, env, camera_pos, camera_target)
    print("📷 VLA 虚拟摄像机架设完毕！")
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, 1, 1))
    cam_pos = gymapi.Vec3(1.5, 0.0, 0.8)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "capture_pose")

    # ==========================================
    # # 🎬 核心剧本组装 (纯净无干涉版)
    # # ==========================================
    # left_p1 = [1.2889, 1.1046, -0.8567, 0.596, 0.0, -0.0, 0.0, -0.0]
    # right_p1 = [1.8796, 1.2171, -0.6129, 0.2095, 0.0, -0.0, 0.0, -0.0]

    # left_p2 = [1.4231, 1.6688, -1.1306, 0.4672, -0.0, 0.0, 0.0481, -0.0481]
    # right_p2 = [1.8796, 1.4426, -0.6135, 0.2095, 0.0, -0.0, 0.05, -0.05]

    # left_p3 = [1.3694, 1.8137, -1.1311, 0.4671, 0.0, -0.0, 0.05, -0.05]
    # right_p3 = [1.8527, 1.8134, -0.9782, 0.0485, 0.0, -0.0, 0.05, -0.05]

    # # 4. 到达抓取位 (悬停在木块两侧)
    # left_p4 = [1.3694, 1.8941, -1.0404, 0.4671, 0.0, -0.0, 0.05, -0.05]
    # right_p4 = [1.8527, 1.8131, -0.8114, 0.0485, 0.0, -0.0, 0.05, -0.05]



    # # 5. 闭合夹取
    # left_p5 = [1.3694, 1.8942, -1.0407, 0.4671, -0.0004, 0.0001, 0.019, -0.02]
    # right_p5 = [1.8529, 1.8131, -0.8118, 0.0484, -0.0019, 0.0001, 0.0176, -0.02]

   
    # # 6. 提拉升空
    # left_p6 = [1.3694, 1.5085, -1.0391, 0.4672, -0.0005, 0.0001, 0.0189, -0.02]
    # right_p6 = [1.8527, 1.5396, -0.8105, 0.0485, -0.0003, 0.0001, 0.0172, -0.02]

    # # 数组组合 (共 8 个关键帧，形成 7 段运动)
    # left_waypoints = [left_p1, left_p2, left_p3, left_p4,  left_p5, left_p6]
    # right_waypoints = [right_p1, right_p2, right_p3, right_p4,  right_p5, right_p6]

    # # ==========================================
    # # ⏱️ 7 段独立变速引擎 (精确操控抓取节奏)
    # # ==========================================
    # transition_times = [
    #     0.5,  # P1 -> P2
    #     0.1,  # P2 -> P3
    #     0.1,  # P3 -> P4
    #     0.5,  # 💥 P4 -> P4_wait: 下探后静止 1.5 秒，绝不乱晃！
    #     0.8,  # P4_wait -> P5: 快速果断闭合夹爪
    #     1.0,  # 💥 P5 -> P5_wait: 夹住后死死等 4 秒！等摩擦力彻底锁死！
    #     0.5   # P5_wait -> P6: 稳稳抬起
    # ] 

    # ==========================================
    # 🎮 播放控制中心
    # ==========================================
    AUTO_PLAY_MODE = False    
    current_waypoint_idx = 0
    
    # 💥 核心修复：舍弃 time.time()，强制使用底层物理时间！
    start_time = gym.get_sim_time(sim)

    print("✅ 全自动分段变速模式已开启！")
    print("⏳ 请欣赏精准抓取大片...")
# ==========================================
    # 💾 初始化 VLA 数据存储目录
    # ==========================================
    dataset_dir = "piper_vla_dataset"
    img_dir = os.path.join(dataset_dir, "images")
    act_dir = os.path.join(dataset_dir, "actions")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(act_dir, exist_ok=True)
    
    frame_count = 0  # 录制帧计数器
    
    # ... 前面的初始化代码保持原样，什么都不用改 ...

    print(f"📁 录像带已备好！数据将存入: ./{dataset_dir}/")
    
    # 从这里开始，替换你原来的 while 循环一直到文件结尾！
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        # ==========================================
        # 🎬 提取相机数组并实时预览
        # ==========================================
        gym.render_all_camera_sensors(sim)
        color_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
        color_image = color_image.reshape((camera_props.height, camera_props.width, 4))
        bgr_image = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_RGB2BGR)
        cv2.imshow("VLA Camera View (Numpy Array)", bgr_image)
        cv2.waitKey(1)
        
        # 📸 快照功能保留
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "capture_pose" and evt.value > 0:
                left_states = gym.get_actor_dof_states(env, left_handle, gymapi.STATE_POS)
                right_states = gym.get_actor_dof_states(env, right_handle, gymapi.STATE_POS)
                print(f"\n🎯 抓拍: 左={[round(float(p), 4) for p in left_states['pos']]}")
                print(f"🎯 抓拍: 右={[round(float(p), 4) for p in right_states['pos']]}\n")

        # 👇==========================================👇
        # 🚀 核心大挪移：网络同步置顶 (彻底移出 AUTO_PLAY_MODE)
        # 👇==========================================👇
        current_time = time.time()
        if current_time - last_send_time >= send_rate:
            if client_conn is None:
                try:
                    # 只要代码在跑，就时刻准备接客！
                    client_conn, addr = server_sock.accept()
                    print(f"✅ 躯干已从隧道接入 {addr}，开始神经同步！")
                except BlockingIOError:
                    pass # 还没有连接，先自己玩，不阻塞渲染
            else:
                # 💥 灵魂改动：不再依赖 AUTO_PLAY_MODE 里的变量
                # 直接强行去抓取仿真环境里此时此刻的目标角度！
                dof_targets = gym.get_actor_dof_position_targets(env, left_handle)
                
                # 提取左臂前 7 个关节的数据
                msg_list = [str(float(x)) for x in dof_targets[:7]]
                msg = ",".join(msg_list) + "\n" # ⚠️ TCP 必须加换行符断句
                
                try:
                    client_conn.sendall(msg.encode('utf-8'))
                except Exception:
                    client_conn.close()
                    client_conn = None
                    print("⚠️ 躯干断开连接，等待重新接入...")
                    
            last_send_time = current_time
        # 👆==========================================👆
        
        # ==========================================
        # 🧠 自动插值引擎与数据采集 (业务逻辑)
        # ==========================================
        if AUTO_PLAY_MODE:
            # 1. 物理时间计算与分段变速
            current_duration = transition_times[current_waypoint_idx]
            t = gym.get_sim_time(sim) - start_time
            alpha = t / current_duration
            
            if alpha >= 1.0:
                alpha = 1.0
                if current_waypoint_idx < len(left_waypoints) - 2:
                    current_waypoint_idx += 1
                    start_time = gym.get_sim_time(sim) # 重新校准物理时间
                    alpha = 0.0
            
            start_pose_l = left_waypoints[current_waypoint_idx]
            end_pose_l = left_waypoints[current_waypoint_idx + 1]
            start_pose_r = right_waypoints[current_waypoint_idx]
            end_pose_r = right_waypoints[current_waypoint_idx + 1]
            
            current_targets_l = [start + (end - start) * alpha for start, end in zip(start_pose_l, end_pose_l)]
            current_targets_r = [start + (end - start) * alpha for start, end in zip(start_pose_r, end_pose_r)]
            
            # 2. 强制对称镜像防打滑
            current_targets_l[7] = -current_targets_l[6]
            current_targets_r[7] = -current_targets_r[6]
            
            gym.set_actor_dof_position_targets(env, left_handle, current_targets_l)
            gym.set_actor_dof_position_targets(env, right_handle, current_targets_r)
            
            # ==========================================
            # 💾 核心：实时落盘存储 (只有在播放时才录制)
            # ==========================================
            if color_image is None or len(color_image) == 0:
                continue
                
            current_action = np.concatenate([current_targets_l, current_targets_r])
            
            img_path = os.path.join(img_dir, f"frame_{frame_count:05d}.jpg")
            act_path = os.path.join(act_dir, f"frame_{frame_count:05d}.npy")
            
            # 写入硬盘 (如果需要存储，把下面两行注释解开)
            # cv2.imwrite(img_path, bgr_image)
            # np.save(act_path, current_action)
            
            import sys
            if frame_count % 50 == 0:
                print(f"⏺️ 正在强力录制... 已完美写入 {frame_count} 帧数据！")
                sys.stdout.flush()
            
            frame_count += 1

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()