import os
import glob
import json
import warnings
from datasets import Dataset, concatenate_datasets

# 屏蔽底层烦人的警告
warnings.filterwarnings('ignore')

print("="*50)
print(" 启动！开始萃取数据...")

# 📁 【核心路径配置区】 (以后换数据集只改这里)
# 1. 你最原始的 data-0000x-of-00005.arrow 碎片存放的文件夹
ARROW_DIR = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert_backup"
# 2. 包含 episodes.jsonl 和 tasks.jsonl 的 meta 文件夹
META_DIR = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert/meta"
# 3. 最终输出完美 parquet 文件的 data 文件夹
OUT_DIR = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert/data"

episodes_file = os.path.join(META_DIR, "episodes.jsonl")

# --- 核心数据重塑逻辑 ---
# 1. 雷达穿透扫描，抓取原始 Arrow 文件
search_pattern = os.path.join(ARROW_DIR, "**", "data-*.arrow")
arrow_files = sorted(glob.glob(search_pattern, recursive=True))
if not arrow_files:
    print(f"找不到原始 Arrow 文件，请检查路径: {ARROW_DIR}")
    exit()

print(f"找到 {len(arrow_files)} 个原始 Arrow 碎片，正在内存中无损拼接...")
ds_list = [Dataset.from_file(f) for f in arrow_files]
ds = concatenate_datasets(ds_list)
print(f" 拼接成功总数据量: {len(ds)} 步！")

# 2. 极速同化：把不符合框架要求的相机名字强转
if 'main_camera' in ds.column_names:
    ds = ds.rename_column('main_camera', 'primary_image')
    print("📸 相机列名 'main_camera' -> 'primary_image' 强转成功！")

# 3. 读取真实的轨迹边界
with open(episodes_file, 'r') as f:
    episodes = [json.loads(line) for line in f]

print(f"正在切割生成 {len(episodes)} 个标准 Parquet ...")
os.makedirs(OUT_DIR, exist_ok=True)

start_idx = 0
for ep in episodes:
    ep_idx = ep['episode_index']
    length = ep['length']
    end_idx = start_idx + length
    
    # 纯血切片，拒绝 Pandas 内存碎片
    ep_ds = ds.select(range(start_idx, end_idx))
    
    # 智能防碰撞：自动补齐框架死磕的身份列
    cols = ep_ds.column_names
    if "task_index" not in cols:
        ep_ds = ep_ds.add_column("task_index", [0] * length)
    if "frame_index" not in cols:
        ep_ds = ep_ds.add_column("frame_index", list(range(length)))
    if "episode_index" not in cols:
        ep_ds = ep_ds.add_column("episode_index", [ep_idx] * length)
    
    # HF 原生导出，彻底规避底层 C++ 段错误
    out_name = f"episode_{ep_idx:06d}.parquet"
    ep_ds.to_parquet(os.path.join(OUT_DIR, out_name))
    start_idx = end_idx
    
    if (ep_idx + 1) % 50 == 0 or (ep_idx + 1) == len(episodes):
        print(f"   进度: [{ep_idx + 1}/{len(episodes)}] 文件安全生成")

# 4. 扫雷行动：清除可能导致误判的旧缓存
pkl_file = os.path.join(META_DIR, "steps_data_index.pkl")
if os.path.exists(pkl_file):
    os.remove(pkl_file)
    print("\n已清除旧版 pkl 缓存！")

print("全部 Parquet 数据集已准备就绪！")
print("="*50)