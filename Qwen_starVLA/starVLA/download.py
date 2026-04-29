from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="StarVLA/Qwen3-VL-OFT-RoboTwin2-All",
#     local_dir="./checkpoints/Qwen3-VL-OFT-RoboTwin2-All", # 指定保存的本地目录
#     resume_download=True,
#     # repo_type 默认为 "model"，所以不需要像数据仓库那样指定 repo_type="dataset"
# )

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct", # 指定保存的本地目录
    resume_download=True,
    # repo_type 默认为 "model"
)

snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    local_dir="./playground/Pretrained_models/Qwen3-VL-4B-Instruct", # 指定保存的本地目录
    resume_download=True,
    # repo_type 默认为 "model"
)