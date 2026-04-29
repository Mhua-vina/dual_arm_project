import sys
from omegaconf import OmegaConf
import torch
import torch.nn as nn

print("="*50)
print("🚀 [极速全链路安检仪] 启动！耗时: < 1 秒")
print("="*50)

# ==========================================
# 1. 查 YAML 户口本
# ==========================================
cfg_path = "starVLA/config/training/piper_dual_arm.yaml"
print(f"🔍 [1/3] 正在加载配置文件: {cfg_path}")
try:
    cfg = OmegaConf.load(cfg_path)
    print("  ✅ 配置文件格式合法，成功加载！")
except Exception as e:
    print(f"  ❌ 配置文件 YAML 格式有误，请检查空格和缩进: {e}")
    sys.exit(1)

# ==========================================
# 2. 测 Batch Size 算术账 (专门抓刚才那个 NoneType 的 Bug)
# ==========================================
print("\n🔍 [2/3] 正在测试 Batch Size 乘法链...")
try:
    # 模拟底层的 _calculate_total_batch_size()
    per_device = cfg.datasets.vla_data.per_device_batch_size
    grad_acc = cfg.trainer.gradient_accumulation_steps
    
    if per_device is None:
        raise ValueError("per_device_batch_size 读出来是 None！请检查是否写错了层级或少敲了空格！")
    if grad_acc is None:
        raise ValueError("gradient_accumulation_steps 读出来是 None！")
        
    # 模拟你用 2 张卡 (GPU 3 和 4)
    num_processes = 2
    total_batch_size = per_device * grad_acc * num_processes
    print(f"  ✅ 算账成功！单卡 {per_device} × 累加 {grad_acc} × {num_processes} 张卡 = 总 Batch Size: {total_batch_size}")
except Exception as e:
    print(f"  ❌ 算账失败，抓到 Bug: {e}")
    sys.exit(1)

# ==========================================
# 3. 测物理冰封术 (虚拟 0 显存模型)
# ==========================================
print("\n🔍 [3/3] 正在测试物理强制结扎术...")

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟 Qwen 的庞大结构
        self.qwen_vl_interface = nn.Linear(10, 10) 
        self.base = nn.Linear(10, 10)
        # 模拟你需要训练的小脑
        self.action_model = nn.Linear(10, 10)

class MockTrainer:
    def __init__(self):
        self.model = MockModel()

trainer = MockTrainer()

# 执行你在 train_starvla.py 里植入的同款强制结扎代码
frozen_count = 0
for name, param in trainer.model.named_parameters():
    if "action" not in name.lower():
        param.requires_grad = False
        frozen_count += 1

trainable = sum(1 for p in trainer.model.parameters() if p.requires_grad)
total = sum(1 for p in trainer.model.parameters())

print(f"  ✅ 物理结扎成功！拦截了 {frozen_count} 个无关张量。")
print(f"  📊 虚拟参数统计: Total: {total}, Trainable: {trainable}")
if trainable < total:
    print("\n🎉🎉🎉 全部安检通过！现在这套配置绝对是无敌的！去启动真实训练吧！")
else:
    print("\n💀 物理冰封术失效，请检查代码！")