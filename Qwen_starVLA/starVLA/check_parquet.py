import pandas as pd
import os

# 指向你刚才生成的 parquet 文件
parquet_path = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset/piper_dual_arm_expert/data/episode_0.parquet"

print("="*50)
print("🚀 正在强行解剖 Parquet 数据文件...")

if not os.path.exists(parquet_path):
    print(f"❌ 完蛋，没找到文件: {parquet_path}")
else:
    df = pd.read_parquet(parquet_path)
    print("✅ Parquet 文件读取成功！")
    print(f"📊 数据总行数 (Frames): {len(df)}")
    
    print("\n🔍 数据集内部真实的列名列表 (Columns):")
    for col in df.columns:
        print(f"  - {col}")
        
    print("\n🎯 精准搜查 'action' 列...")
    if "action" in df.columns:
        sample = df["action"].iloc[0]
        print(f"  ✅ 完美！找到了 'action' 列！")
        print(f"  📦 数据类型: {type(sample)}")
        print(f"  📝 第一行动作样例: {sample}")
    else:
        print("  🚨 破案了！你的数据里根本没有名叫 'action' 的列！")
        suspects = [c for c in df.columns if 'act' in c.lower()]
        print(f"  🤔 疑似动作的列名可能是: {suspects}")

print("="*50)