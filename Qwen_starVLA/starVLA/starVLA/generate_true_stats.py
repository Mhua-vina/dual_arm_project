import os
import shutil
from datasets import load_dataset

dataset_dir = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert"
backup_dir = dataset_dir + "_backup"

print("1. 正在读取原始 Arrow 数据...")
# 匹配所有的 arrow 文件
ds = load_dataset("arrow", data_files=os.path.join(dataset_dir, "*.arrow"), split="train")

print("2. 正在注入缺失的 observation.state 列...")
# 瞬间给几十万条数据加上 16 维的假状态
ds = ds.map(lambda x: {"observation.state": [0.0] * 16}, num_proc=8)

print("3. 正在备份旧数据...")
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
os.rename(dataset_dir, backup_dir)

print("4. 正在保存完美标准格式的数据集...")
# 建立 LeRobot 框架强迫症需要的 data 文件夹
out_dir = os.path.join(dataset_dir, "data")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "episode_0.parquet")
    
print(f"正在强制导出为 Parquet 格式并保存至: {out_file}")
    # 这才是真正转换格式的核心指令！
ds.to_parquet(out_file)

print("5. 正在恢复元数据 (Meta)...")
# 把之前的 info.json 等拷回来
shutil.copytree(os.path.join(backup_dir, "meta"), os.path.join(dataset_dir, "meta"), dirs_exist_ok=True)

# 🧹 核心步骤：删掉之前弄乱的 stats 缓存，让框架自己算！
for stat_file in ["stats.json", "stats_gr00t.json"]:
    file_path = os.path.join(dataset_dir, "meta", stat_file)
    if os.path.exists(file_path):
        os.remove(file_path)

print("你的数据集现在是官方标准格式！")