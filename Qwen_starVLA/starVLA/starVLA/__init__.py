# 文件位置: /mnt/bigdata/minghua/starVLA/vla_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import base64
from io import BytesIO
from PIL import Image

# ！！！这里需要替换为你实际加载 StarVLA 模型的代码 ！！！
# from starVLA.model import load_model 
# model = load_model("path_to_your_weights")
print("🚀 [大脑节点] StarVLA 模型加载中...")

app = FastAPI()

# 定义接收的数据结构
class Payload(BaseModel):
    image: str
    instruction: str

@app.post("/predict")
def predict_action(data: Payload):
    print(f"📥 [大脑节点] 收到指令: {data.instruction}")
    
    # 1. 把 Base64 字符串还原回图片
    img_data = base64.b64decode(data.image)
    image = Image.open(BytesIO(img_data))
    
    # 2. 将图片和指令喂给模型 (替换为真实推理代码)
    # action_tensor = model.predict(image, data.instruction)
    # action_list = action_tensor.tolist()
    
    # ⚠️ 下面是假数据，用于打通链路测试。
    # 假设机械臂有 14 个动作维度 (双臂各7个)
    mock_action = [0.0] * 14 
    
    print("📤 [大脑节点] 推理完成，动作已下发！")
    return {"action": mock_action}

if __name__ == "__main__":
    # 启动服务器，绑定到 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)