import sys
from omegaconf import OmegaConf

# 把当前路径加入系统路径，防止找不到包
sys.path.append(".")

# 1. 直接导入你刚才疯狂报错的那个核心函数
from starVLA.dataloader import build_dataloader

# 2. 读取你的 YAML 配置文件
cfg_path = "starVLA/config/training/piper_dual_arm.yaml"
cfg = OmegaConf.load(cfg_path)

print("正在绕过模型，直接暴力测试数据集加载...")

try:
    # 3. 单刀直入，只运行数据集组装逻辑
    # 💥 强行撬开 OmegaConf 的结构锁，塞入日志输出目录！
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = "./test_outputs"
    
    # 原本的代码保持不变
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    print("✅ 数据管道彻底打通，没有任何报错！")
    print(f"📦 数据集样本概览: {dataloader}")
except Exception as e:
    print("❌ 抓到 Bug 了！报错信息如下：")
    raise e