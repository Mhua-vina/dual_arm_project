"""
快速诊断脚本：验证冻结逻辑是否正常工作
"""
import sys
sys.path.insert(0, '/mnt/data01/minghua/Qwen_starVLA/starVLA')

from omegaconf import OmegaConf
import torch.nn as nn

# 加载配置
cfg = OmegaConf.load("starVLA/config/training/piper_dual_arm.yaml")
print("="*60)
print("配置加载成功！")
print(f"冻结模块列表: {cfg.trainer.freeze_modules}")
print("="*60)

# 模拟模型结构 (QwenAdapter 真实结构)
class MockQwenVLInterface(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟 Qwen 的大层结构
        self.model = nn.ModuleDict({
            'layers': nn.ModuleList([nn.Linear(100, 100) for _ in range(10)])
        })

class MockActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(16, 16)

class MockStarVLA(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟 QwenAdapter 的两个子模块
        self.qwen_vl_interface = MockQwenVLInterface()
        self.action_model = MockActionModel()

model = MockStarVLA()

# 统计冻结前
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📊 冻结前: Total={total_params}, Trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 导入并执行冻结
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

freeze_modules = cfg.trainer.freeze_modules
print(f"\n🔒 正在冻结模块: {freeze_modules}")
model = TrainerUtils.freeze_backbones(model, freeze_modules=freeze_modules)

# 统计冻结后
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"\n📊 冻结后:")
print(f"  - 可训练参数: {trainable}")
print(f"  - 冻结参数: {frozen}")
print(f"  - 可训练比例: {trainable/total_params*100:.2f}%")

if trainable < total_params:
    print("\n✅ 冻结逻辑验证通过！")
else:
    print("\n❌ 冻结失败！请检查冻结逻辑。")
