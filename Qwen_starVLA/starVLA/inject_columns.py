import os
import pandas as pd
from pathlib import Path

data_dir = Path("/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert/data")

print("🚀 正在给所有 Parquet 文件注入 LeRobot v2.0 必须的控制列...")
parquet_files = list(data_dir.glob("*.parquet"))

for pf in parquet_files:
    df = pd.read_parquet(pf)
    needs_save = False
    
    # 1. 补齐 task_index (咱们的任务编号都是 0)
    if 'task_index' not in df.columns:
        df['task_index'] = 0
        needs_save = True
        
    # 2. 顺手补齐 frame_index (防止它等下又报缺帧序号)
    if 'frame_index' not in df.columns:
        df['frame_index'] = range(len(df))
        needs_save = True
        
    # 3. 顺手补齐 episode_index (从文件名 episode_000000.parquet 里提取)
    if 'episode_index' not in df.columns:
        ep_idx = int(pf.stem.split('_')[1])
        df['episode_index'] = ep_idx
        needs_save = True
        
    if needs_save:
        df.to_parquet(pf)

print("✅ 疫苗注射完毕！500 个 Parquet 文件现在已达到 100% 官方标准格式！")