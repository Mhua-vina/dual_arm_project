import socket
import math
import time

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    print("❌ 找不到 piper_sdk...")
    exit()

print("⏳ 正在唤醒物理机械臂...")
try:
    piper = C_PiperInterface_V2(can_name="0", can_auto_init=False)
    piper.CreateCanBus(can_name="0", bustype="gs_usb", expected_bitrate=1000000)
    piper.ConnectPort()
    while(not piper.EnablePiper()): time.sleep(0.01)
    print("✅ 物理机械臂已挂载刚度！")
except Exception as e:
    print("❌ 机械臂连接失败，检查线缆并 sudo 运行！")
    exit()

# ==========================================
# 建立 TCP 客户端，通过隧道连接 4090
# ==========================================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("📡 正在通过本地 SSH 隧道入口寻找 4090 大脑...")
while True:
    try:
        # 注意！这里直接连本机的 8888！隧道会负责把它搬运到 4090
        sock.connect(("127.0.0.1", 8888))
        print("✅ 成功连接大脑！准备接收指令...")
        break
    except ConnectionRefusedError:
        time.sleep(1) # 如果 4090 还没启动，就耐心等待

last_gripper_pulse = -1
buffer = ""

try:
    while True:
        # TCP 流式接收，按换行符 \n 拆解动作
        data = sock.recv(1024).decode('utf-8')
        if not data: break
        buffer += data
        while "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            try:
                vals = [float(x) for x in msg.split(',')]
                if len(vals) >= 7:
                    piper.MotionCtrl_2(0x01, 0x01, 100)
                    scale = 1000 
                    j = [int(vals[i] * (180 / math.pi) * scale) for i in range(6)]
                    piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
                    
                    finger_val = abs(vals[6])
                    max_sim_open = 0.04 
                    max_real_pulse = 80000 
                    open_ratio = min(finger_val / max_sim_open, 1.0)
                    target_pulse = int(open_ratio * max_real_pulse)
                    
                    if abs(target_pulse - last_gripper_pulse) > 1000:
                        piper.GripperCtrl(target_pulse, 5000, 0x01, 0x00)
                        last_gripper_pulse = target_pulse
            except Exception:
                pass # 忽略解析错误

except KeyboardInterrupt:
    print("\n🛑 接收到中断信号！")
finally:
    piper.DisablePiper()
    sock.close()
    print("💤 机械臂已安全休眠！")