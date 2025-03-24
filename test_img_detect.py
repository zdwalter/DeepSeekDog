import torch
import cv2
import sys
from pathlib import Path
from typing import Union, List, Dict

class OfflineYOLODetector:
    """离线YOLOv5目标检测器"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = None
        self._required_files = [
            'hubconf.py',
            'models/common.py',
            'models/yolo.py',
            'utils/general.py'
        ]
    
    def _verify_local_repo(self, repo_path: Union[str, Path]) -> bool:
        """验证本地YOLOv5仓库完整性"""
        repo_path = Path(repo_path)
        missing = [f for f in self._required_files if not (repo_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"YOLOv5仓库不完整，缺失文件：{missing}\n"
                "请执行：git clone https://github.com/ultralytics/yolov5")
        return True

    def init_model(
        self,
        local_repo: str = "yolov5",
        model_weights: str = "yolov5s.pt",
        force_cpu: bool = False
    ) -> None:
        """
        初始化离线模型
        :param local_repo: 本地YOLOv5代码仓库路径
        :param model_weights: 本地模型权重文件路径
        :param force_cpu: 强制使用CPU模式
        """
        try:
            # 检查基础依赖
            self._check_dependencies()
            
            # 验证本地仓库
            self._verify_local_repo(local_repo)
            
            # 验证权重文件
            weights_path = Path(model_weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"模型权重文件不存在：{weights_path}")
            
            # 设置计算设备
            self.device = self._get_device(force_cpu)
            print(f"🖥️  运行设备：{self.device}")
            
            # 加载本地模型
            self.model = torch.hub.load(
                str(local_repo),
                'custom',
                source='local',
                path=str(weights_path),
                autoshape=True,
                verbose=False,
                skip_validation=True  # 跳过在线校验
            ).to(self.device)
            
            # 加载元数据
            self.class_names = self.model.names
            print(f"✅ 离线模型加载成功（支持{len(self.class_names)}个类别）")
            
        except Exception as e:
            print(f"❌ 模型初始化失败：{str(e)}")
            sys.exit(1)
    
    def _get_device(self, force_cpu: bool) -> torch.device:
        """获取可用计算设备"""
        if force_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("⚠️  未检测到CUDA设备，使用CPU模式")
        return torch.device('cpu')
    
    def _check_dependencies(self) -> None:
        """检查必要依赖"""
        required = {'torch', 'torchvision', 'numpy', 'cv2'}
        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                f"缺少必要依赖：{missing}\n"
                "请执行：pip install " + " ".join(missing))
    
    def detect(
        self,
        img_path: Union[str, Path],
        conf_thresh: float = 0.5,
        output_path: Union[str, Path] = None
    ) -> Dict:
        """
        执行目标检测
        :param img_path: 输入图像路径
        :param conf_thresh: 置信度阈值（0-1）
        :param output_path: 输出文件路径（支持.jpg/.png/.txt）
        :return: 检测结果字典
        """
        if not self.model:
            raise RuntimeError("模型未初始化，请先调用init_model()")
        
        try:
            # 读取输入图像
            img_path = Path(img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"输入图片不存在：{img_path}")
            
            # 执行推理
            results = self.model(img_path, size=640)
            
            # 解析结果
            detections = results.pandas().xyxy[0]
            valid_detections = detections[detections.confidence >= conf_thresh]
            
            # 生成输出
            output = {
                "image_size": (results.ims[0].shape[1], results.ims[0].shape[0]),
                "detections": [],
                "objects_count": len(valid_detections),
                "success": True
            }
            
            # 构建检测结果
            for _, row in valid_detections.iterrows():
                output["detections"].append({
                    "class": row['name'],
                    "confidence": round(row['confidence'], 4),
                    "bbox": (
                        int(row['xmin']), int(row['ymin']),
                        int(row['xmax']), int(row['ymax'])
                    )
                })
            
            # 保存输出文件
            if output_path:
                output_path = Path(output_path)
                self._save_output(results, output_path)
                output["output_path"] = str(output_path.absolute())
            
            return output
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_output(self, results, output_path: Path) -> None:
        """保存输出结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            # 保存可视化图像
            results.render()  # 添加标注框
            cv2.imwrite(str(output_path), cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR))
            print(f"📸 检测结果已保存至：{output_path}")
            
        elif output_path.suffix == '.txt':
            # 保存文本报告
            with open(output_path, 'w') as f:
                f.write(f"检测对象总数：{len(results.xyxy[0])}\n")
                for detection in results.xyxy[0]:
                    class_id = int(detection[5])
                    f.write(
                        f"{self.class_names[class_id]} "
                        f"({detection[4]:.2f}) | "
                        f"坐标：{int(detection[0])},{int(detection[1])} "
                        f"{int(detection[2])},{int(detection[3])}\n"
                    )
            print(f"📝 文本报告已保存至：{output_path}")
        
        else:
            raise ValueError("不支持的输出格式，请使用.jpg/.png/.txt")

if __name__ == "__main__":
    # 使用示例
    detector = OfflineYOLODetector()
    
    # 初始化模型（假设本地仓库在./yolov5目录）
    detector.init_model(
        local_repo="yolov5",
        model_weights="yolov5s.pt",
        force_cpu=False  # 强制使用CPU
    )
    
    # 执行检测
    result = detector.detect(
        img_path="test.jpg",
        conf_thresh=0.6,
        output_path="detection_result.jpg"
    )
    
    if result['success']:
        print(f"🎯 检测到{result['objects_count']}个对象：")
        for obj in result['detections']:
            print(f" - {obj['class']} ({obj['confidence']:.2f})")
    else:
        print(f"❌ 检测失败：{result['error']}")
