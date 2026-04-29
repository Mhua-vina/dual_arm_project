import base64
import io
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

# ==========================================
# 🌟 核心填空 1：导入你的 VLA 模型类
# ==========================================
# 去看一眼 starVLA/examples/ 下的测试脚本是怎么 import 的
# 通常长这样（如果报错找不到模块，请告诉我 starVLA 目录下的具体结构）：
from starVLA.model.vla_model import VLAModel # <--- 请根据实际代码里的类名修改
# 如果有单独的 Processor（预处理器），也一并导入，例如：
# from starVLA.model.processor import VLAProcessor 

app = FastAPI()

# 全局变量，用来在显存中“常驻”这个大模型，防止每次请求都重新加载
VLA_BRAIN = None

def load_vla_model():
    """
    启动服务时，把几 GB 的大模型载入 4090 的显存中待命。
    """
    global VLA_BRAIN
    print("⏳ [系统] 正在将 starVLA 模型载入显卡，大概需要几分钟...")
    
    # ==========================================
    # 🌟 核心填空 2：填写你的模型权重路径
    # ==========================================
    # 指向你目录里的 weights 文件夹下的具体 .pt 或 .bin 文件
    model_weight_path = "/mnt/bigdata/minghua/starVLA_project/weights/your_model_weight.pth" 
    
    # 根据 starVLA 的实际 API 加载模型（下面是三种最常见的范式，选一种没被注释的用）
    
    # 范式 A (最常见)：直接从预训练路径加载
    VLA_BRAIN = VLAModel.from_pretrained(model_weight_path).to("cuda")
    
    # 范式 B：先实例化，再 load_state_dict
    # VLA_BRAIN = VLAModel()
    # VLA_BRAIN.load_state_dict(torch.load(model_weight_path, map_location="cuda"))
    # VLA_BRAIN.to("cuda")
    
    # 设置为评估模式，关闭梯度计算，省一半显存！
    VLA_BRAIN.eval()
    print("✅ [系统] 大脑载入完毕！随时可以开始控制躯体。")

class VLARequest(BaseModel):
    image_base64: str
    instruction: str

@app.post("/predict")
async def predict_action(request: VLARequest):
    if VLA_BRAIN is None:
        return {"status": "error", "message": "模型未加载完成"}

    try:
        # 1. 解析 Isaac Gym 发来的图片
        image_data = base64.b64decode(request.image_base64)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        instruction = request.instruction
        
        print(f"📥 [收到请求] 指令: '{instruction}'")

        # ==========================================
        # 🌟 核心填空 3：执行真正的推理 (Inference)
        # ==========================================
        # 大多数 VLA 模型为了防止爆显存，推理时需要加上 torch.no_grad()
        with torch.no_grad():
            # 这里调用 starVLA 真正的预测函数。
            # 函数名可能是 predict, inference, 或 generate_action
            
            # 伪代码示例：
            # inputs = processor(image_pil, instruction).to("cuda")
            # action_tensor = VLA_BRAIN.predict_action(**inputs)
            
            # 由于目前我没看到 starVLA 的内部代码，这里假定它叫 predict：
            action_tensor = VLA_BRAIN.predict(image_pil, instruction)
            
        # 2. 将张量 (Tensor) 转换为 Python 列表，方便通过网络发回给仿真
        # 假设 action_tensor 的形状是 [7] (6D位姿 + 1D夹爪)
        action_list = action_tensor.cpu().numpy().tolist()
        
        # 如果模型吐出来的是二维数组 [[x,y,z...]]，把它拍扁：
        if isinstance(action_list[0], list):
            action_list = action_list[0]
            
        print(f"📤 [决策输出] 动作向量: {action_list}")

        return {"status": "success", "action": action_list}

    except Exception as e:
        print(f"❌ [推理崩溃] 报错详情: {e}")
        return {"status": "error", "message": str(e)}

# 注册启动事件：当 FastAPI 启动时，自动触发模型加载
@app.on_event("startup")
async def startup_event():
    load_vla_model()

if __name__ == "__main__":
    # 使用 Uvicorn 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)