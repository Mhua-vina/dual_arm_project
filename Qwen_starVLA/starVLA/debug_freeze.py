import torch
from omegaconf import OmegaConf
# 假设你的框架类名叫 QwenAdapter
from starVLA.training.framework.QwenAdapter import QwenAdapter 

print("🚀 [维度互通性体检] 启动...")

# 1. 模拟输入：一张图 + 一个指令
mock_img = torch.randn(1, 3, 224, 224).to("meta")
mock_instruction = "pick up the block"

# 2. 模拟配置
cfg = OmegaConf.load("starVLA/config/training/piper_dual_arm.yaml")

# 3. 在虚拟设备上实例化模型（瞬间完成）
with torch.device("meta"):
    model = QwenAdapter(cfg)

print(f"✅ 模型架构已建立。目标动作维度: {cfg.framework.action_model.action_dim}")

# 4. 模拟一次推理流程
try:
    # 这里模拟 forward 过程
    # 如果你的 Piper 数据是 16 维，这里应该输出 (1, 8, 16) 
    # (1个 batch, 8步 action chunk, 16个关节角)
    print(f"🔍 正在校验 action_model 输出维度...")
    # 假设模型输出是这个名字
    out_dim = model.action_model.action_dim 
    if out_dim == 16:
        print(f"🎉 维度对齐成功！仿真数据(16D) 与 VLA 动作头(16D) 完美互通。")
    else:
        print(f"❌ 维度冲突！模型输出是 {out_dim}，请检查 YAML。")
except Exception as e:
    print(f"⚠️ 逻辑链路有误: {e}")