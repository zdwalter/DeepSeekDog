import torch
import cv2
from pathlib import Path

class ImageDetector:
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = None

    def init_model(self):
        """初始化目标检测模型
        依赖：torch, torchvision, opencv-python, matplotlib, pyyaml
        """
        try:
            # 设备检测
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备：{self.device}")

            # 加载YOLOv5模型
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                      pretrained=True, 
                                      autoshape=True,
                                      verbose=False).to(self.device)
            
            self.model.eval()  # 设置为评估模式
            self.class_names = self.model.names  # 获取类别名称
            print(f"模型初始化完成，支持检测{len(self.class_names)}种对象")

        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            self.model = None

    def detect_image(self, img_path, output_path="output.jpg", conf_thresh=0.5):
        """执行图像检测并保存结果
        :param img_path: 输入图片路径
        :param output_path: 输出文件路径（支持.jpg/.png/.txt）
        :param conf_thresh: 置信度阈值（0-1）
        :return: 检测结果字典
        """
        if not self.model:
            print("错误：模型未初始化")
            return None

        try:
            # 读取图像
            if not Path(img_path).exists():
                raise FileNotFoundError(f"图片文件不存在: {img_path}")
            
            # 执行推理
            results = self.model(img_path, size=640)  # 调整推理尺寸
            
            # 解析结果
            detections = results.pandas().xyxy[0]  # 转换为DataFrame
            valid_detections = detections[detections.confidence >= conf_thresh]
            
            # 生成输出
            output = {
                "image_size": (results.ims[0].shape[1], results.ims[0].shape[0]),
                "detections": [],
                "output_path": str(Path(output_path).absolute())
            }

            # 保存可视化结果（图片格式）
            if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                results.render()  # 添加标注框
                cv2.imwrite(output_path, cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR))
                print(f"结果已保存至：{output_path}")
            
            # 保存文本报告
            elif output_path.endswith('.txt'):
                with open(output_path, 'w') as f:
                    f.write(f"检测结果（置信度阈值：{conf_thresh}）\n")
                    f.write(f"图像尺寸：{output['image_size']}\n")
                    for _, row in valid_detections.iterrows():
                        f.write(f"{row['name']}: {row['confidence']:.2f} @ {row['xmin']:.0f},{row['ymin']:.0f},{row['xmax']:.0f},{row['ymax']:.0f}\n")
                print(f"文本报告已保存至：{output_path}")

            # 构建返回数据
            for _, row in valid_detections.iterrows():
                output["detections"].append({
                    "class": row['name'],
                    "confidence": round(row['confidence'], 2),
                    "bbox": (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                })

            return output

        except Exception as e:
            print(f"检测失败：{str(e)}")
            return None

# 使用示例
if __name__ == "__main__":
    detector = ImageDetector()
    detector.init_model()
    
    if detector.model:
        result = detector.detect_image(
            img_path="test.jpg",
            output_path="detection_result.jpg",
            conf_thresh=0.6
        )
        
        if result:
            print(f"检测到{len(result['detections'])}个对象：")
            for obj in result['detections']:
                print(f"- {obj['class']} ({obj['confidence']})")
