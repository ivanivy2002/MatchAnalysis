from ultralytics import YOLO
import torch

# 检查CUDA是否可用，并选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 初始化模型并加载到相应设备
# model = YOLO('yolov8x')
model = YOLO('models/best.pt')
model.to(device)

# 在选择的设备上运行预测
result = model.predict('input_video/08fd33_4.mp4', save=True, device=device)

# 打印预测结果
print(result[0])
print('************************')
for box in result[0].boxes:
    print(box)

# 转为xyxy标签
