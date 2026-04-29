import torch
import torch.nn as nn
from omegaconf import OmegaConf

print("="*50)
print(" [极速全链路安检仪 v2.0 - 纯净 Adapter 版] 启动！")
print("="*50)

# ==========================================
# 1. 模拟你的全套模型架构 (用 0 显存的 nn.Module 替代)
# ==========================================
class MockQwenVLInterface(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟那个恐怖的 900 亿参数大脑
        self.visual = nn.Linear(10, 1000)
        self.model = nn.ModuleDict({
            'layers': nn.Linear(10, 1000),
            'embed_tokens': nn.Embedding(10, 10)
        })
        self.lm_head = nn.Linear(10, 1000)

class MockActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟你要训练的小脑
        self.mlp = nn.Linear(16, 16)

class MockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 5)

class MockStarVLA(nn.Module):
    def __init__(self):
        super().__init__()
        # 这就是日志里打印出来的三大主组件
        self.qwen_vl_interface = MockQwenVLInterface()
        self.action_model = MockActionModel()
        self.base = MockBase()

class MockTrainer:
    def __init__(self):
        self.model = MockStarVLA()

trainer = MockTrainer()

# ==========================================
# 2. 执行真正的精准结扎术
# ==========================================
print("❄️ 物理结...")
for name, param in trainer.model.named_parameters():
    # 只要名字里带有 qwen, visual, base, lm_head 这些大脑组件的字眼，统统锁死！
    if any(k in name.lower() for k in ["qwen", "visual", "base", "lm_head"]):
        param.requires_grad = False
    else:
        # 只有纯正的下游小脑（Adapter）允许训练
        param.requires_grad = True

# ==========================================
# 3. 生成体检报告
# ==========================================
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())

print("\n📊 真实手术报告:")
print(f"  总参数: {total} (模拟值)")
print(f"  可训练参数: {trainable} (模拟值)")

# 判断逻辑：因为你的小脑很小，如果结扎成功，可训练参数占比一定会极低
ratio = trainable / total
print(f"  可训练比例: {ratio*100:.2f}%")

if ratio < 0.1: # 占比不到 10%
    print("\n大脑已被彻底冻结，你的 24G 显卡稳了！赶紧去跑真实的训练吧！")
else:
    print("\n还有大量参数未被冻结！请检查上述代码里的关键词是否匹配到了所有的层。")