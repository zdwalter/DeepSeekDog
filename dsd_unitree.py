#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
#  依赖库导入
# ============================================================
import asyncio
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import edge_tts
import numpy as np
import pyaudio
import speech_recognition as sr
import torch
from flask import Flask, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
from vosk import KaldiRecognizer, Model

# ---------- 新增：Matplotlib 用于点云可视化 ----------
import matplotlib
matplotlib.use("Agg")          # 无显示环境也能保存图片
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (触发 3D 投影注册)

# ---------- Unitree‑SDK ----------
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
    ObstaclesAvoidClient,
)

# ============================================================
#  常量
# ============================================================
TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

# ============================================================
#  Lidar 处理类
# ============================================================
class LidarProcessor:
    """LIDAR 数据处理"""

    def __init__(self):
        self.latest_cloud_data = None
        self.lidar_running = False
        self.lidar_thread = None
        self.cloud_subscriber = None
        self.obstacle_distance = None
        self.scan_complete = False

    # ---------- 启动 / 停止 ----------
    def start_lidar(self):
        if self.lidar_running:
            return
        self.cloud_subscriber = ChannelSubscriber(TOPIC_CLOUD, PointCloud2_)
        self.cloud_subscriber.Init(handler=self.point_cloud_callback)

        self.lidar_running = True
        self.lidar_thread = threading.Thread(
            target=self._lidar_worker, daemon=True
        )
        self.lidar_thread.start()
        print("LIDAR 数据接收已启动")

    def stop_lidar(self):
        if not self.lidar_running:
            return
        self.lidar_running = False
        if self.cloud_subscriber:
            self.cloud_subscriber.Close()
        if self.lidar_thread:
            self.lidar_thread.join(timeout=1)
        print("LIDAR 数据接收已停止")

    # ---------- 订阅回调 ----------
    def point_cloud_callback(self, cloud_data: PointCloud2_):
        if cloud_data is not None:
            self.latest_cloud_data = cloud_data
            self._process_point_cloud(cloud_data)

    # ---------- 点云处理 ----------
    def _process_point_cloud(self, cloud_data: PointCloud2_):
        try:
            # 1. Data validation
            if not cloud_data.fields or len(cloud_data.data) == 0:
                raise ValueError("Empty point cloud data")
    
            # 2. Create ASCII grid
            grid_cols, grid_rows = 60, 20
            grid = [["·" for _ in range(grid_cols)] for _ in range(grid_rows)]
    
            # 3. Convert sequence[uint8] to bytes
            cloud_bytes = bytes(cloud_data.data)  # Convert list of uint8 to bytes
    
            # 4. Parse field configurations
            field_config = {}
            byte_order = "big" if cloud_data.is_bigendian else "little"
            
            for field in cloud_data.fields:
                dtype = {
                    1: ("uint8", 1),
                    2: ("int8", 1),
                    3: ("uint16", 2),
                    4: ("int16", 2),
                    5: ("uint32", 4),
                    6: ("int32", 4),
                    7: ("float32", 4),
                    8: ("float64", 8),
                }.get(field.datatype)
                if not dtype:
                    raise ValueError(f"Unsupported field type: {field.datatype}")
    
                field_config[field.name.lower()] = {
                    "offset": field.offset,
                    "dtype": dtype[0],
                    "size": dtype[1],
                    "count": field.count,
                }
    
            # 5. Verify required fields exist
            for axis in ["x", "y", "z"]:
                if axis not in field_config:
                    raise ValueError(f"Missing coordinate field: {axis}")
    
            # 6. Process points
            valid_points = []
            for i in range(0, len(cloud_bytes), cloud_data.point_step):
                point = {}
                for axis in ["x", "y", "z"]:
                    cfg = field_config[axis]
                    start = i + cfg["offset"]
                    end = start + cfg["size"]
                    chunk = cloud_bytes[start:end]
    
                    if "float" in cfg["dtype"]:
                        value = np.frombuffer(chunk, dtype=cfg["dtype"])[0]
                    else:
                        value = int.from_bytes(
                            chunk,
                            byteorder=byte_order,
                            signed=cfg["dtype"].startswith("i"),
                        )
                    point[axis] = float(value)
    
                valid_points.append(point)
    
            # 7. Save 3D point cloud visualization
            self._save_point_cloud_image(valid_points)
    
    
            # 9. Update state
            self.obstacle_distance = (
                min(p['x'] for p in valid_points) if valid_points else None
            )
            self.scan_complete = True
    
        except Exception as exc:
            print(f"点云处理异常: {type(exc).__name__}: {exc}")
            self.obstacle_distance = None
            self.scan_complete = False

    # ---------- 新增：点云图保存 ----------
    def _save_point_cloud_image(self, points):
        """将当前点云保存为 point_cloud.jpg（覆盖式）
        根据点与原点的距离显示不同颜色，按0.5米间隔：
        - 0-0.5m: 红色
        - 0.5-1.0m: 橙色
        - 1.0-1.5m: 黄色
        - 1.5-2.0m: 黄绿色
        - 2.0-2.5m: 绿色
        - 2.5-3.0m: 蓝绿色
        - 3.0m+: 蓝色
        """
        if not points:
            return
        try:
            arr = np.array([[p["x"], p["y"], p["z"]] for p in points])
            
            # 计算每个点到原点的距离
            distances = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
            
            # 定义颜色映射 (每0.5米一个颜色)
            color_map = [
                (0.0, 0.5, [1.0, 0.0, 0.0]),    # 红色 (0-0.5m)
                (0.5, 1.0, [1.0, 0.5, 0.0]),    # 橙色
                (1.0, 1.5, [1.0, 1.0, 0.0]),    # 黄色
                (1.5, 2.0, [0.7, 1.0, 0.0]),    # 黄绿色
                (2.0, 2.5, [0.0, 1.0, 0.0]),   # 绿色
                (2.5, 3.0, [0.0, 1.0, 0.7]),    # 蓝绿色
                (3.0, float('inf'), [0.0, 0.0, 1.0])  # 蓝色 (3m+)
            ]
            
            # 为每个点分配颜色
            colors = []
            for d in distances:
                for min_d, max_d, color in color_map:
                    if min_d <= d < max_d:
                        colors.append(color)
                        break
            
            # 创建3D图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            
            # 绘制点云，使用不同颜色
            ax.scatter(
                arr[:, 0],  # X坐标
                arr[:, 1],  # Y坐标
                arr[:, 2],  # Z坐标
                c=colors,    # 颜色数组
                s=2,         # 点大小
                alpha=0.7    # 透明度
            )
            
            # 设置坐标轴标签和标题
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("Point Cloud (Color by 0.5m Distance Intervals)")
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, label=f'{min_d}-{max_d if max_d != float("inf") else "∞"}m')
                for min_d, max_d, color in color_map
            ]
            
            ax.legend(
                handles=legend_elements, 
                loc='upper right',
                title="Distance Range",
                bbox_to_anchor=(1.25, 1.0)
            
            # 调整视角和布局
            ax.view_init(elev=25, azim=45)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig("static/photos/point_cloud.jpg", dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as exc:
            print(f"点云图片保存失败: {exc}")

    # ---------- 后台线程 ----------
    def _lidar_worker(self):
        while self.lidar_running:
            time.sleep(0.1)

    # ---------- 外部接口 ----------
    def scan_environment(self) -> bool:
        self.scan_complete = False
        start = time.time()
        while not self.scan_complete and time.time() - start < 3.0:
            time.sleep(0.1)
        return self.scan_complete

    def get_obstacle_distance(self):
        return self.obstacle_distance

    def get_point_cloud_info(self):
        if not self.latest_cloud_data:
            return None
        return {
            "height": self.latest_cloud_data.height,
            "width": self.latest_cloud_data.width,
            "point_step": self.latest_cloud_data.point_step,
            "row_step": self.latest_cloud_data.row_step,
            "is_dense": self.latest_cloud_data.is_dense,
            "data_size": len(self.latest_cloud_data.data),
            "timestamp": time.time(),
        }


# ============================================================
#  Web 接口
# ============================================================
class WebInterface:
    def __init__(self, voice_controller, host="0.0.0.0", port=5000):
        self.app = Flask(
            __name__, template_folder="templates", static_folder="static"
        )
        self.socketio = SocketIO(self.app)
        self.voice_controller = voice_controller
        self.host = host
        self.port = port

        self.auto_photo_enabled = True
        self.photo_interval = 3000  # ms

        # ---------- 路由 ----------
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule(
            "/photos", "get_photos", self.get_photos, methods=["GET"]
        )
        self.app.add_url_rule("/photo/<filename>", "get_photo", self.get_photo)
        self.app.add_url_rule(
            "/status", "get_status", self.get_status, methods=["GET"]
        )
        self.app.add_url_rule(
            "/lidar", "get_lidar", self.get_lidar, methods=["GET"]
        )

        os.makedirs("static/photos", exist_ok=True)

        # ---------- 自动拍照线程 ----------
        self._start_auto_photo()

        # ---------- Web 服务器线程 ----------
        self.server_thread = threading.Thread(
            target=self.run_server, daemon=True
        )
        self.server_thread.start()

        # ---------- Socket.IO ----------
        @self.socketio.on("user_message")
        def handle_user_message(data):
            msg = data.get("message", "").strip()
            if msg:
                self.voice_controller.cmd_queue.put(msg)

        @self.socketio.on("take_photo")
        def handle_take_photo():
            self.voice_controller.cmd_queue.put("拍照")

        @self.socketio.on("command")
        def handle_command(data):
            cmd = data.get("command", "").strip()
            if cmd:
                self.voice_controller.cmd_queue.put(cmd)

    # ---------- 自动拍照 ----------
    def _start_auto_photo(self):
        if self.auto_photo_enabled:
            threading.Thread(
                target=self._auto_photo_worker, daemon=True
            ).start()
            print(f"自动拍照已启用，默认间隔: {self.photo_interval}ms")

    def _auto_photo_worker(self):
        while self.auto_photo_enabled and hasattr(self, "voice_controller"):
            start = time.time()
            try:
                self.voice_controller.handle_take_photo()
                elapsed = (time.time() - start) * 1000
                time.sleep(max(0, self.photo_interval - elapsed) / 1000)
            except Exception as exc:
                print(f"自动拍照错误: {exc}")
                time.sleep(1)

    # ---------- Socket 消息 ----------
    def send_tts_message(self, message):
        self.socketio.emit("tts_message", {"message": message})

    # ---------- HTTP ----------
    def index(self):
        return render_template("index.html")

    def get_photos(self):
        photos = []
        photo_path = "static/photos/photo.jpg"
        if os.path.exists(photo_path):
            photos.append(
                {
                    "filename": "photo.jpg",
                    "timestamp": os.path.getmtime(photo_path),
                    "url": "/photo/photo.jpg",
                }
            )
        return jsonify(sorted(photos, key=lambda x: x["timestamp"], reverse=True))

    def get_photo(self, filename):
        return send_from_directory("static/photos", filename)

    def get_lidar(self):
        if hasattr(self.voice_controller, "lidar_processor"):
            info = self.voice_controller.lidar_processor.get_point_cloud_info()
            if info:
                return jsonify(info)
        return jsonify({"error": "LIDAR data not available"}), 404

    def get_status(self):
        return jsonify(
            {
                "running": self.voice_controller.running,
                "network": self.voice_controller.current_network_status,
                "mode": "debug"
                if self.voice_controller.debug_mode
                else "normal",
                "lidar": "active"
                if self.voice_controller.lidar_processor.lidar_running
                else "inactive",
            }
        )

    # ---------- run ----------
    def run_server(self):
        print(f"Starting web server on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port)

    def save_photo(self, image_data):
        filename = "photo.jpg"
        path = f"static/photos/{filename}"
        with open(path, "wb") as f:
            f.write(image_data)

        self.socketio.emit(
            "new_photo",
            {
                "filename": filename,
                "timestamp": time.time(),
                "url": f"/photo/{filename}",
            },
        )
        return path


# ============================================================
#  YOLOv5 离线检测器
# ============================================================
class OfflineYOLODetector:
    """离线 YOLOv5 目标检测器"""

    def __init__(self, repo_path="yolov5", model_weights="yolov5s.pt"):
        self.model = None
        self.class_names = None
        print("开始加载 YOLOv5 模型...")
        self._init_detector(repo_path, model_weights)
        print("YOLOv5 模型加载完成")

        # 英文->中文标签
        self.label_translation = {
            "person": "人",
            "car": "汽车",
            "chair": "椅子",
            "book": "书",
            "cell phone": "手机",
            "cup": "杯子",
            "laptop": "笔记本电脑",
            "dog": "狗",
            "cat": "猫",
            "bottle": "瓶子",
            "keyboard": "键盘",
            "mouse": "鼠标",
            "tv": "电视",
            "umbrella": "雨伞",
            "backpack": "背包",
        }

    # ---------- 初始化 ----------
    def _verify_paths(self, repo_path, model_weights):
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError("YOLOv5 代码仓库不存在")
        for req in ["hubconf.py", "models/common.py"]:
            if not (repo_path / req).exists():
                raise FileNotFoundError(f"缺失文件: {req}")
        if not Path(model_weights).exists():
            raise FileNotFoundError("模型权重文件不存在")

    def _init_detector(self, repo_path, model_weights):
        try:
            self._verify_paths(repo_path, model_weights)
            self.model = torch.hub.load(
                str(repo_path),
                "custom",
                source="local",
                path=str(model_weights),
                autoshape=True,
                verbose=True,
                skip_validation=True,
            )
            self.class_names = self.model.names
        except Exception as exc:
            print(f"模型初始化失败: {exc}")
            self.model = None

    # ---------- 推理 ----------
    def analyze(self, image_path, conf_thresh=0.5):
        if not self.model:
            return None
        try:
            results = self.model(image_path, size=320)
            detections = results.pandas().xyxy[0]
            valid = detections[detections.confidence >= conf_thresh]

            items = []
            for _, row in valid.iterrows():
                cn = self.label_translation.get(row["name"].lower(), row["name"])
                if cn not in items:
                    items.append(cn)
            return items
        except Exception as exc:
            print(f"识别错误: {exc}")
            return None


# ============================================================
#  语音控制主类
# ============================================================
class VoiceControl:
    def __init__(self):
        # ---------- 信号 ----------
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # ---------- 队列 ----------
        self.cmd_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.asr_queue = queue.Queue()

        # ---------- 异步 ----------
        self.loop = asyncio.new_event_loop()
        self.tts_thread = None

        # ---------- 状态 ----------
        self.running = True
        self.debug_mode = True

        # ---------- 检测器 ----------
        self.detector = OfflineYOLODetector(
            repo_path="yolov5", model_weights="yolov5s.pt"
        )
        self.last_photo_path = ""

        # ---------- 网络 ----------
        self.current_network_status = self.check_network()

        # ---------- Web ----------
        self.web_interface = WebInterface(self)

        # ---------- 音频 ----------
        self.audio_interface = pyaudio.PyAudio()
        self.init_voice_engine()

        # ---------- 视频 ----------
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

        # ---------- Lidar ----------
        self.lidar_processor = LidarProcessor()
        self.lidar_processor.start_lidar()

        # ---------- 命令映射 ----------
        self.command_map = {
            r"停(止|下)?$": 6,
            r"stop$": 6,
            r"前(进|行)$": 3,
            r"forward$": 3,
            r"后(退|撤)$": 3,
            r"back(ward)?$": 3,
            r"左移$": 4,
            r"left$": 4,
            r"右移$": 4,
            r"right$": 4,
            r"旋(转|转)$": 5,
            r"turn$": 5,
            r"站(立|起)$": 1,
            r"stand$": 1,
            r"坐(下|姿)$": 2,
            r"sit$": 2,
            r"平衡站$": 9,
            r"balance$": 9,
            r"调试模式$": -1,
            r"正常模式$": -2,
            r"退出$": 0,
            r"拍(照|照片|图片)$": 7,
            r"take photo$": 7,
            r"识别(物品|内容)$": 8,
            r"analyze$": 8,
            r"扫描环境$": 10,
            r"scan environment$": 10,
            r"障碍物检测$": 11,
            r"obstacle detection$": 11,
            r"前方障碍物$": 12,
            r"front obstacle$": 12,
        }

        # ---------- 运动 ----------
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        # ---------- 避障 ----------
        self.obstacle_client = ObstaclesAvoidClient()
        self.obstacle_client.SetTimeout(10.0)
        self.obstacle_client.Init()
        self.obstacle_client.SwitchSet(True)
        self.obstacle_client.UseRemoteCommandFromApi(True)

        print("初始化完成\n")

    # --------------------------------------------------------
    #  TTS 协程
    # --------------------------------------------------------
    async def tts_worker(self):
        while self.running:
            try:
                text = self.tts_queue.get_nowait()
                if text:
                    print(f"[TTS] {text}")
                    self.web_interface.send_tts_message(text)

                    communicate = edge_tts.Communicate(
                        text=text,
                        voice="zh-CN-YunxiNeural",
                        rate="+10%",
                        volume="+0%",
                    )
                    player = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "-"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            player.stdin.write(chunk["data"])
                    player.stdin.close()
                    await asyncio.sleep(0.1)
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as exc:
                print(f"TTS 错误: {exc}")
                self.web_interface.send_tts_message(f"语音输出错误: {exc}")

    # --------------------------------------------------------
    #  信号处理
    # --------------------------------------------------------
    def signal_handler(self, signum, frame):
        print("\n收到终止信号，正在退出...")
        self.running = False
        self.cleanup_resources()

    # --------------------------------------------------------
    #  语音设备初始化
    # --------------------------------------------------------
    def init_voice_engine(self):
        try:
            self.vosk_model = Model(lang="cn")
            self.online_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=2048,
                stream_callback=self.audio_callback,
                start=False,
            )
            self.audio_stream.start_stream()
            print("语音设备初始化成功")
        except Exception as exc:
            self.cleanup_resources()
            raise RuntimeError(f"语音设备初始化失败: {exc}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.running:
            self.asr_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
        return (None, pyaudio.paComplete)

    # --------------------------------------------------------
    #  网络状态
    # --------------------------------------------------------
    def check_network(self):
        return False  # Debug：不检查网络
        try:
            socket.create_connection(("www.baidu.com", 80), timeout=2)
            return True
        except OSError:
            return False

    # --------------------------------------------------------
    #  ASR：在线
    # --------------------------------------------------------
    def online_asr_processing(self):
        with self.microphone as source:
            self.online_recognizer.adjust_for_ambient_noise(source)
            while self.running and self.current_network_status:
                try:
                    audio = self.online_recognizer.listen(source, timeout=3)
                    text = self.online_recognizer.recognize_google(
                        audio, language="zh-CN"
                    )
                    print(f"[在线识别] {text}")
                    self.cmd_queue.put(text)
                except sr.UnknownValueError:
                    pass
                except sr.WaitTimeoutError:
                    continue
                except sr.RequestError:
                    self.current_network_status = False
                    self.tts_queue.put("网络连接中断，切换至离线模式")

    # --------------------------------------------------------
    #  ASR：离线
    # --------------------------------------------------------
    def offline_asr_processing(self):
        rec = KaldiRecognizer(self.vosk_model, 16000)
        while self.running and not self.current_network_status:
            data = self.asr_queue.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"[离线识别] {text}")
                    self.cmd_queue.put(text)

    # --------------------------------------------------------
    #  网络监控
    # --------------------------------------------------------
    def network_monitor(self):
        while self.running:
            new_status = self.check_network()
            if new_status != self.current_network_status:
                self.current_network_status = new_status
                self.tts_queue.put(
                    f"网络状态已切换至{'在线' if new_status else '离线'}模式"
                )
            time.sleep(5)

    # --------------------------------------------------------
    #  命令解析
    # --------------------------------------------------------
    def process_command(self, command):
        cmd = command.lower().replace(" ", "")
        print(f"{cmd}")

        # 模式切换
        if re.search(r"调试模式$", cmd):
            self.debug_mode = True
            self.tts_queue.put("已进入调试模式")
            return
        if re.search(r"正常模式$", cmd):
            self.debug_mode = False
            self.tts_queue.put("已进入正常模式")
            return
        if re.search(r"退出$", cmd):
            self.running = False
            self.tts_queue.put("正在关机")
            return

        # 匹配命令
        action_id = None
        for pattern, cmd_id in self.command_map.items():
            if re.search(pattern, cmd):
                action_id = cmd_id
                break

        if action_id is not None:
            self.execute_action(action_id, cmd)
        else:
            self.tts_queue.put("指令未识别")

    # --------------------------------------------------------
    #  执行动作
    # --------------------------------------------------------
    def execute_action(self, action_id, raw_cmd=""):
        print(f"执行动作 {action_id} ({raw_cmd})")
        try:
            # ---------- Lidar 特殊动作 ----------
            if action_id == 10:  # 扫描环境
                if self.lidar_processor.scan_environment():
                    info = self.lidar_processor.get_point_cloud_info()
                    self.tts_queue.put(
                        f"环境扫描完成，获取 {info['data_size']} 个数据点"
                    )
                else:
                    self.tts_queue.put("环境扫描失败")
                return

            if action_id == 11:  # 障碍物检测
                if self.lidar_processor.scan_environment():
                    dist = self.lidar_processor.get_obstacle_distance()
                    if dist is not None:
                        self.tts_queue.put(f"前方 {dist:.1f} 米处检测到障碍物")
                    else:
                        self.tts_queue.put("前方未检测到障碍物")
                else:
                    self.tts_queue.put("障碍物检测失败")
                return

            if action_id == 12:  # 实时查询
                dist = self.lidar_processor.get_obstacle_distance()
                if dist is not None:
                    self.tts_queue.put(f"前方 {dist:.1f} 米处有障碍物")
                else:
                    self.tts_queue.put("前方暂无障碍物")
                return

            # ---------- 运动控制 ----------
            if action_id == 6:  # 停止
                self.sport_client.StopMove()

            elif action_id == 3:  # 前进 / 后退
                speed = -0.3 if any(x in raw_cmd for x in ["后", "back"]) else 0.3
                self.sport_client.Move(speed, 0, 0)

            elif action_id == 4:  # 左 / 右
                speed = -0.3 if any(x in raw_cmd for x in ["右", "right"]) else 0.3
                self.sport_client.Move(0, speed, 0)

            elif action_id == 5:  # 旋转
                self.sport_client.Move(0, 0, 0.5)

            elif action_id == 1:  # 站立
                self.sport_client.Euler(0.1, 0.2, 0.3)
                self.sport_client.BodyHeight(0.0)
                self.sport_client.BalanceStand()

            elif action_id == 2:  # 坐下
                self.sport_client.StandDown()

            elif action_id == 9:  # 平衡站
                self.sport_client.BalanceStand()

            elif action_id == 7:  # 拍照
                self.handle_take_photo()

            elif action_id == 8:  # 分析图片
                self.handle_image_analysis()

            if not self.debug_mode and action_id not in (7, 8, 10, 11, 12):
                self.tts_queue.put("指令已执行")

        except Exception as exc:
            err = f"执行出错: {exc}"
            self.tts_queue.put(err)
            print(err)

    # --------------------------------------------------------
    #  拍照
    # --------------------------------------------------------
    def handle_take_photo(self):
        code, data = self.video_client.GetImageSample()
        if code == 0:
            self.last_photo_path = self.web_interface.save_photo(bytes(data))
            self.tts_queue.put("拍照成功")
            self.handle_image_analysis()
        else:
            self.tts_queue.put("拍照失败，请重试")

    # --------------------------------------------------------
    #  图像分析
    # --------------------------------------------------------
    def handle_image_analysis(self):
        if not self.last_photo_path:
            self.tts_queue.put("请先拍摄照片")
            return
        if not os.path.exists(self.last_photo_path):
            self.tts_queue.put("照片文件不存在")
            return

        items = self.detector.analyze(self.last_photo_path)
        if items:
            result = "、".join(items[:3])
            self.tts_queue.put(f"识别到: {result}")
        else:
            self.tts_queue.put("未识别到有效物品")

    # --------------------------------------------------------
    #  资源清理
    # --------------------------------------------------------
    def cleanup_resources(self):
        print("正在清理资源...")
        self.running = False

        # 音频
        if hasattr(self, "audio_stream") and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        if hasattr(self, "audio_stream"):
            self.audio_stream.close()
        if hasattr(self, "audio_interface"):
            self.audio_interface.terminate()

        # 运动
        try:
            self.sport_client.StandDown()
            time.sleep(2)
            self.sport_client.StopMove()
        except Exception:
            pass

        # 视频
        try:
            del self.video_client
        except Exception:
            pass

        # Lidar
        try:
            self.lidar_processor.stop_lidar()
        except Exception:
            pass

        # YOLO
        if self.detector and self.detector.model:
            del self.detector.model
            torch.cuda.empty_cache()

        # 事件循环
        if self.loop.is_running():
            self.loop.stop()

        # 清空队列
        for q in (self.tts_queue, self.asr_queue):
            while not q.empty():
                q.get_nowait()

    # --------------------------------------------------------
    #  主循环
    # --------------------------------------------------------
    def run(self):
        try:
            self.execute_action(1)  # 先平衡站立
            self.tts_queue.put("系统已启动，准备就绪")

            # TTS 线程
            self.tts_thread = threading.Thread(
                target=self.loop.run_forever, daemon=True
            )
            self.tts_thread.start()
            asyncio.run_coroutine_threadsafe(self.tts_worker(), self.loop)

            # 其他线程
            threads = [
                threading.Thread(target=self.network_monitor, daemon=True),
                threading.Thread(target=self.offline_asr_processing, daemon=True),
            ]
            if self.current_network_status:
                threads.append(
                    threading.Thread(
                        target=self.online_asr_processing, daemon=True
                    )
                )
            for t in threads:
                t.start()

            # 主命令循环
            while self.running:
                try:
                    command = self.cmd_queue.get(timeout=0.5)
                    self.process_command(command)
                except queue.Empty:
                    continue
        finally:
            self.cleanup_resources()
            print("程序已安全退出")


# ============================================================
#  主入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <networkInterface>")
        sys.exit(-1)

    ChannelFactoryInitialize(0, sys.argv[1])

    try:
        controller = VoiceControl()
        controller.run()
    except Exception as exc:
        print(f"系统错误: {exc}")
    finally:
        print("程序退出")
