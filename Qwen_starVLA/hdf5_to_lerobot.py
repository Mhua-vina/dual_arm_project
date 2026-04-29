import os
import h5py
from datasets import Dataset, Features, Sequence, Value, Image as DatasetsImage
from PIL import Image

def main():
    # 你的“生鲜”原材料路径
    hdf5_path = "/mnt/data01/minghua/Qwen_starVLA/piper_dual_arm_expert.hdf5"
    # 切好的“净菜”保存路径
    out_dir = "/mnt/data01/minghua/Qwen_starVLA/starVLA/dataset/piper_lerobot_dataset"
    
    # 防止重复写入
    if os.path.exists(out_dir):
        print(f"⚠️ 目标文件夹 {out_dir} 已存在，请先删除或改名！")
        return

    def gen_data():
        """这是一个生成器，用来一行行地把 HDF5 数据喂给 HuggingFace / LeRobot 底层"""
        with h5py.File(hdf5_path, 'r') as f:
            data_group = f['data']
            episodes = list(data_group.keys())
            
            print(f"📦 发现 {len(episodes)} 个专家回合数据，准备开始切片...")
            
            for ep_idx, ep_name in enumerate(episodes):
                ep_data = data_group[ep_name]
                images = ep_data['images'][:]
                actions = ep_data['actions'][:]
                instruction = ep_data.attrs.get('language_instruction', 'Grasp the wood block with both arms.')
                
                num_frames = len(actions)
                for frame_idx in range(num_frames):
                    # 把 numpy 数组转回图片格式
                    img_array = images[frame_idx]
                    pil_img = Image.fromarray(img_array)
                    
                    # 严格按照 LeRobot 要求的 Schema 吐出数据
                    yield {
                        "observation.images.main_camera": pil_img,
                        "action": actions[frame_idx][:14].tolist(), # 强制截断，只保留 14 维双臂动作                            "task": instruction,
                        "task": instruction,
                        "episode_index": ep_idx,
                        "frame_index": frame_idx,
                        "timestamp": frame_idx * 0.1  # 假定控制频率是 10Hz
                    }

    # 严丝合缝地定义 LeRobot 格式 (Schema)
    features = Features({
        "observation.images.main_camera": DatasetsImage(),
        "action": Sequence(Value("float32")),
        "task": Value("string"),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "timestamp": Value("float32")
    })

    print("🚀 启动数据切菜机，开始压缩并写入 Parquet 格式...")
    # 这个函数会自动利用多核 CPU 把你的数据压缩成工业级标准！
    dataset = Dataset.from_generator(gen_data, features=features)
    dataset.save_to_disk(out_dir)
    print(f"🎉 转换彻底完成！你可以用这个数据集去喂 VLA 大模型了！")

if __name__ == "__main__":
    main()