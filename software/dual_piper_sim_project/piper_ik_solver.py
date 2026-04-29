import pybullet as p
import numpy as np

class PiperIKSolver:
    def __init__(self, urdf_path, ee_link_index=6):
        # 💥 核心：以 DIRECT 模式启动，不弹任何 GUI 窗口，纯做数学运算，速度极快
        self.physics_client = p.connect(p.DIRECT)
        
        # 加载跟你 Isaac Gym 里一模一样的机械臂图纸
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        self.ee_index = ee_link_index
        
    def solve(self, target_pos, target_quat):
        """
        输入目标 XYZ 坐标和四元数姿态，输出完美的 6 轴关节角度
        """
        # 调用内置的 DLS 阻尼最小二乘法求解器
        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_index,
            targetPosition=target_pos,
            targetOrientation=target_quat,
            maxNumIterations=200,      # 迭代次数给足，保证精度
            residualThreshold=1e-5     # 误差阈值卡死在 0.01 毫米级
        )
        
        # 提取前 6 个大臂关节的角度（过滤掉夹爪和固定轴）
        return list(joint_angles[:6])
        
    def close(self):
        p.disconnect(self.physics_client)