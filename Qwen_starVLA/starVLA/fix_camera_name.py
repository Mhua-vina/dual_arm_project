import os
import glob
import json
import warnings
from datasets import Dataset, concatenate_datasets

# 屏蔽底层烦人的警告
warnings.filterwarnings('ignore')

print("="*50)
print("🚀 [完美闭环] 左手拿纯净 Arrow，右手拿跨服 JSONL，智能防撞融合！")

arrow_dir = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert_backup"
meta_dir = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert/meta"
out_dir = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert/data"

episodes_file = os.path.join(meta_dir, "episodes.jsonl")

# --- 核心逻辑开始 ---
search_pattern = os.path.join(arrow_dir, "**", "data-*.arrow")
arrow_files = sorted(glob.glob(search_pattern, recursive=True))

print(f"🔍 成功揪出 {len(arrow_files)} 个原始 Arrow 文件，开始内存无损熔炼...")

ds_list = [Dataset.from_file(f) for f in arrow_files]
ds = concatenate_datasets(ds_list)
print(f"✅ 拼接成功！总数据量: {len(ds)} 步！")

# 极速改名
if 'main_camera' in ds.column_names:
    ds = ds.rename_column('main_camera', 'primary_image')
    print("📸 相机列名 'main_camera' -> 'primary_image' 强转成功！")

# 读取切分边界
with open(episodes_file, 'r') as f:
    episodes = [json.loads(line) for line in f]

print(f"🔪 正在切割生成 {len(episodes)} 个完美 Parquet (带智能防撞护盾)...")
os.makedirs(out_dir, exist_ok=True)

start_idx = 0
for ep in episodes:
    ep_idx = ep['episode_index']
    length = ep['length']
    end_idx = start_idx + length
    
    ep_ds = ds.select(range(start_idx, end_idx))
    
    # 🛡️ 智能防碰撞：原来这就叫画蛇添足！如果有就不加，没有才加！
    cols = ep_ds.column_names
    if "task_index" not in cols:
        ep_ds = ep_ds.add_column("task_index", [0] * length)
    if "frame_index" not in cols:
        ep_ds = ep_ds.add_column("frame_index", list(range(length)))
    if "episode_index" not in cols:
        ep_ds = ep_ds.add_column("episode_index", [ep_idx] * length)
    
    out_name = f"episode_{ep_idx:06d}.parquet"
    ep_ds.to_parquet(os.path.join(out_dir, out_name))
    
    start_idx = end_idx
    
    if (ep_idx + 1) % 50 == 0 or (ep_idx + 1) == len(episodes):
        print(f"  ⏳ 进度: [{ep_idx + 1}/{len(episodes)}] 文件安全生成！")

# 打扫战场
pkl_file = os.path.join(meta_dir, "steps_data_index.pkl")
if os.path.exists(pkl_file):
    os.remove(pkl_file)
    print("\n🧹 已清除带毒的 pkl 缓存！")

print("🎉 融合完毕！最完美的 500 个文件已经躺在 data 文件夹里了！")
print("="*50)