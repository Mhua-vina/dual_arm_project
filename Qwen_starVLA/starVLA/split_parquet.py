import pandas as pd
import json
import os

base_dir = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert"
mega_parquet = os.path.join(base_dir, "data", "episode_0.parquet")
episodes_file = os.path.join(base_dir, "meta", "episodes.jsonl")
info_file = os.path.join(base_dir, "meta", "info.json")

print("1. 🚀 正在读取咱们之前生成的巨无霸 Parquet...")
df = pd.read_parquet(mega_parquet)

print("2. 📖 正在读取你搬运过来的 episodes.jsonl 边界信息...")
with open(episodes_file, 'r') as f:
    episodes = [json.loads(line) for line in f]

print(f"3. 🔪 正在将巨无霸精准切割成 {len(episodes)} 个独立的物理文件...")
start_idx = 0
for ep in episodes:
    ep_idx = ep['episode_index']
    length = ep['length']
    end_idx = start_idx + length
    
    # 按照真实的边界切分数据
    ep_df = df.iloc[start_idx:end_idx]
    
    # 构造官方标准的六位数字文件名
    out_name = f"episode_{ep_idx:06d}.parquet"
    out_path = os.path.join(base_dir, "data", out_name)
    ep_df.to_parquet(out_path)
    
    start_idx = end_idx

print("4. 🛠️ 修复 info.json 中的框架寻址模式...")
with open(info_file, 'r') as f:
    info = json.load(f)
# 让框架用正确的变量名去寻找刚才切出来的 500 个文件
info['data_path'] = 'data/episode_{episode_index:06d}.parquet'
with open(info_file, 'w') as f:
    json.dump(info, f, indent=4)

print("✅ 完美！物理文件和元数据终于彻底对齐了！")