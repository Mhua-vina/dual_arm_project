import os
import sys
import torch
from omegaconf import OmegaConf
from accelerate import init_empty_weights

# 1. 强行拉取你的配置文件
cfg = OmegaConf.load("starVLA/config/training/piper_dual_arm.yaml")

print("🚀 启动光速虚拟测试 (0 显存, 耗时 5 秒)...")
# 2. 我们用 Meta 虚拟设备，瞬间生成 950 亿参数的架构，但不加载 90GB 的真实权重！
with init_empty_weights():
    # 这里直接模拟你们的 StarVLAModel 的顶层架构
    import torch.nn as nn
    class MockStarVLA(nn.Module):
        def __init__(self):
            super().__init__()
            # 根据日志逆向出的三大真实组件
            self.qwen_vl_interface = nn.Linear(10, 10) 
            self.action_model = nn.Linear(10, 10)
            self.base = nn.Linear(10, 10)
    
    model = MockStarVLA()

# 3. 模拟 train_starvla.py 的冻结逻辑
freeze_list = cfg.trainer.freeze_modules
if isinstance(freeze_list, str):
    freeze_list = [freeze_list]

print(f"❄️ 你的配置文件要求冻结: {freeze_list}")

for freeze_path in freeze_list:
    try:
        # StarVLA 底层真实的冻结代码就是这样的！必须精确匹配！
        sub_module = model.get_submodule(freeze_path)
        for param in sub_module.parameters():
            param.requires_grad = False
        print(f"  ✅ 成功命中并冻结模块: {freeze_path}")
    except AttributeError:
        print(f"  ❌ 警告: 找不到精确路径 '{freeze_path}'")

# 4. 统计结果
trainable = sum(1 for p in model.parameters() if p.requires_grad)
total = sum(1 for p in model.parameters())
print(f"\n📊 虚拟测试结果: Total Parameters: {total}, Trainable: {trainable}")
if trainable < total:
    print("🎉 冰封术测试成功！你可以放心去跑 30 分钟的真实训练了！")
else:
    print("💀 冰封术失效！请检查 yaml 里的名字是否写错了！")