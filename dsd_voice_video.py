import time
import sys
import threading
import queue
import socket
import json
import re
import speech_recognition as sr
import edge_tts
import asyncio
import subprocess
import os
import pyaudio
import cv2
import torch
import numpy as np
from vosk import Model, KaldiRecognizer
from pathlib import Path
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

import signal

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import threading
import os
import time


class WebInterface:
    def __init__(self, voice_controller, host='0.0.0.0', port=5000):
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.socketio = SocketIO(self.app)
        self.voice_controller = voice_controller
        self.host = host
        self.port = port
        
        # 设置路由
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/photos', 'get_photos', self.get_photos, methods=['GET'])
        self.app.add_url_rule('/photo/<filename>', 'get_photo', self.get_photo)
        self.app.add_url_rule('/command', 'send_command', self.send_command, methods=['POST'])
        self.app.add_url_rule('/status', 'get_status', self.get_status, methods=['GET'])
        
        # 创建必要的目录
        os.makedirs('static/photos', exist_ok=True)
        
        # 启动Web服务器线程
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
    
    def index(self):
        return render_template('index.html')
    
    def get_photos(self):
        photos = []
        photo_path = 'static/photos/photo.jpg'
        if os.path.exists(photo_path):
            photos.append({
                'filename': 'photo.jpg',
                'timestamp': os.path.getmtime(photo_path),
                'url': '/photo/photo.jpg'
            })
        return jsonify(sorted(photos, key=lambda x: x['timestamp'], reverse=True))
    
    def get_photo(self, filename):
        return send_from_directory('static/photos', filename)
    
    def send_command(self):
        command = request.json.get('command')
        if command:
            self.voice_controller.cmd_queue.put(command)
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'No command provided'}), 400
    
    def get_status(self):
        return jsonify({
            'running': self.voice_controller.running,
            'network': self.voice_controller.current_network_status,
            'mode': 'debug' if self.voice_controller.debug_mode else 'normal'
        })
    
    def run_server(self):
        print(f"Starting web server on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port)
    
    def save_photo(self, image_data):
        filename = "photo.jpg"
        path = f"static/photos/{filename}"
        with open(path, 'wb') as f:
            f.write(image_data)
        
        # 通知所有客户端有新照片
        self.socketio.emit('new_photo', {
            'filename': filename,
            'timestamp': time.time(),
            'url': f'/photo/{filename}'
        })
        return path
        
class OfflineYOLODetector:
    """离线YOLOv5目标检测器（优化版）"""
    
    def __init__(self, repo_path="yolov5", model_weights="yolov5s.pt"):
        self.model = None
        self.class_names = None
        print("开始加载YOLOv5模型...")
        self._init_detector(repo_path, model_weights)
        print("YOLOv5模型加载完成")
        
        # 中英标签映射
        self.label_translation = {
            'person': '人', 'car': '汽车', 'chair': '椅子',
            'book': '书', 'cell phone': '手机', 'cup': '杯子',
            'laptop': '笔记本电脑', 'dog': '狗', 'cat': '猫',
            'bottle': '瓶子', 'keyboard': '键盘', 'mouse': '鼠标',
            'tv': '电视', 'umbrella': '雨伞', 'backpack': '背包'
        }
        
    def _init_detector(self, repo_path, model_weights):
        """初始化检测器（带错误处理）"""
        try:
            self._verify_paths(repo_path, model_weights)
            self.model = torch.hub.load(
                str(repo_path),
                'custom',
                source='local',
                path=str(model_weights),
                autoshape=True,
                verbose=True,
                skip_validation=True
            )
            self.class_names = self.model.names
            print("图像识别模型加载成功")
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            self.model = None

    def _verify_paths(self, repo_path, model_weights):
        """验证模型路径"""
        required_files = ['hubconf.py', 'models/common.py']
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError("YOLOv5代码仓库不存在")
        for f in required_files:
            if not (repo_path / f).exists():
                raise FileNotFoundError(f"缺失文件: {f}")
        if not Path(model_weights).exists():
            raise FileNotFoundError("模型权重文件不存在")

    def analyze(self, image_path, conf_thresh=0.5):
        """执行图像分析"""
        if not self.model:
            return None
            
        try:
            results = self.model(image_path, size=320)  # 降低分辨率提升速度
            detections = results.pandas().xyxy[0]
            valid_detections = detections[detections.confidence >= conf_thresh]
            
            items = []
            for _, row in valid_detections.iterrows():
                cn_name = self.label_translation.get(row['name'].lower(), row['name'])
                if cn_name not in items:
                    items.append(cn_name)
            return items
        except Exception as e:
            print(f"识别错误: {str(e)}")
            return None

class VoiceControl:
    def __init__(self):
        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 队列系统
        self.cmd_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.asr_queue = queue.Queue()

        # 异步事件循环
        self.loop = asyncio.new_event_loop()
        self.tts_thread = None

        # 状态控制
        self.running = True
        self.debug_mode = True  # 默认调试模式
        
        # 图像识别初始化（修改部分）
        self.detector = OfflineYOLODetector(repo_path="yolov5", model_weights="yolov5s.pt")
        self.last_photo_path = ""
        

        # 网络状态
        self.current_network_status = self.check_network()

        self.web_interface = WebInterface(self)
        
        # 音频设备初始化
        self.audio_interface = pyaudio.PyAudio()
        self.init_voice_engine()

        
        # 视频客户端初始化
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()



        # 命令映射表
        self.command_map = {
            r"停(止|下)?$": 6, r"stop$": 6,
            r"前(进|行)$": 3, r"forward$": 3,
            r"后(退|撤)$": 3, r"back(ward)?$": 3,
            r"左移$": 4, r"left$": 4,
            r"右移$": 4, r"right$": 4,
            r"旋(转|转)$": 5, r"turn$": 5,
            r"站(立|起)$": 1, r"stand$": 1,
            r"坐(下|姿)$": 2, r"sit$": 2,
            r"平衡站$": 9, r"balance$": 9,
            r"调试模式$": -1,
            r"正常模式$": -2,
            r"退出$": 0,
            r"拍(照|照片|图片)$": 7, r"take photo$": 7,
            r"识别(物品|内容)$": 8, r"analyze$": 8
        }
        
        # 运动控制初始化
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # 添加避障客户端初始化
        self.obstacle_client = ObstaclesAvoidClient()
        self.obstacle_client.SetTimeout(10.0)
        self.obstacle_client.Init()
        
        # 启用避障模式
        self.obstacle_client.SwitchSet(True)
        self.obstacle_client.UseRemoteCommandFromApi(True)

        


        print("初始化完成\n")
        



            
    def init_image_recognition(self):
        """初始化图像识别模型"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()
            print("图像识别模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None

    async def tts_worker(self):
        """TTS工作协程"""
        while self.running:
            try:
                text = self.tts_queue.get_nowait()
                if text:
                    print(f"[TTS] {text}")
                    communicate = edge_tts.Communicate(
                        text=text,
                        voice="zh-CN-YunxiNeural",
                        rate="+10%", 
                        volume="+0%"
                    )
                    
                    player = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "-"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            player.stdin.write(chunk["data"])
                    player.stdin.close()
                    await asyncio.sleep(0.1)
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"TTS错误: {str(e)}")

    def signal_handler(self, signum, frame):
        """信号处理"""
        print("\n收到终止信号，正在退出...")
        self.running = False
        self.cleanup_resources()

    def init_voice_engine(self):
        """语音设备初始化"""
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
                start=False
            )
            self.audio_stream.start_stream()
            print("语音设备初始化 成功")
        except Exception as e:
            self.cleanup_resources()
            raise RuntimeError(f"语音设备初始化失败: {str(e)}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调"""
        if self.running:
            self.asr_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
        else:
            return (None, pyaudio.paComplete)

    def check_network(self):
        """网络检测"""
        return False  # debug模式
        try:
            socket.create_connection(("www.baidu.com", 80), timeout=2)
            return True
        except OSError:
            return False

    def online_asr_processing(self):
        """在线语音识别"""
        with self.microphone as source:
            self.online_recognizer.adjust_for_ambient_noise(source)
            while self.running and self.current_network_status:
                try:
                    audio = self.online_recognizer.listen(source, timeout=3)
                    text = self.online_recognizer.recognize_google(audio, language='zh-CN')
                    print(f"[在线识别] {text}")
                    self.cmd_queue.put(text)
                except sr.UnknownValueError:
                    print("无法识别语音")
                except sr.WaitTimeoutError:
                    continue
                except sr.RequestError:
                    self.current_network_status = False
                    self.tts_queue.put("网络连接中断，切换至离线模式")

    def offline_asr_processing(self):
        """离线语音识别"""
        rec = KaldiRecognizer(self.vosk_model, 16000)
        while self.running and not self.current_network_status:
            data = self.asr_queue.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get('text', '')
                if text:
                    print(f"[离线识别] {text}")
                    self.cmd_queue.put(text)

    def network_monitor(self):
        """网络状态监测"""
        while self.running:
            new_status = self.check_network()
            if new_status != self.current_network_status:
                self.current_network_status = new_status
                status = "在线" if new_status else "离线"
                self.tts_queue.put(f"网络状态已切换至{status}模式")
            time.sleep(5)

    def process_command(self, command):
        """命令处理"""
        cmd = command.lower().replace(" ", "")
        print(f"{cmd}")
        
        # 移除启动确认检查
        if re.search(r"调试模式$", cmd):
            self.debug_mode = True
            self.tts_queue.put("已进入调试模式")
            return
        elif re.search(r"正常模式$", cmd):
            self.debug_mode = False
            self.tts_queue.put("已进入正常模式")
            return
        elif re.search(r"退出$", cmd):
            self.running = False
            self.tts_queue.put("正在关机")
            return
    
        action_id = None
        for pattern, cmd_id in self.command_map.items():
            if re.search(pattern, cmd):
                action_id = cmd_id
                print(f"{pattern}, {cmd}, {action_id}")
                break
    
        if action_id is not None:
            self.execute_action(action_id, cmd)
        else:
            self.tts_queue.put("指令未识别")

    def execute_action(self, action_id, raw_cmd=""):
        """执行动作（添加障碍物检测）"""
        print(f"执行动作{raw_cmd}, {action_id}")
        try:
            if 0:
                if action_id in [3, 4]:  # 前进或左右移动
                    # 检查前方障碍物
                    obstacle = self.get_obstacle_info()
                    
                    if obstacle and obstacle['distance'] < 0.5:  # 0.5米内有障碍物
                        self.tts_queue.put(f"前方{obstacle['distance']:.1f}米处检测到障碍物")
                        return
                    
            if action_id == 6:
                self.sport_client.StopMove()
            elif action_id == 3:
                speed = -0.3 if any(x in raw_cmd for x in ["后", "back"]) else 0.3
                # 使用避障模式移动
                self.sport_client.Move(speed, 0, 0)
            elif action_id == 4:
                speed = -0.3 if any(x in raw_cmd for x in ["右", "right"]) else 0.3
                # 使用避障模式移动
                self.sport_client.Move(0, speed, 0)
            elif action_id == 5:
                # 旋转仍然使用普通模式
                self.sport_client.Move(0, 0, 0.5)
            elif action_id == 1:
                self.sport_client.Euler(0.1, 0.2, 0.3)
                self.sport_client.BodyHeight(0.0)
                self.sport_client.BalanceStand()
            elif action_id == 2:
                self.sport_client.StandDown()
            elif action_id == 9:
                self.sport_client.BalanceStand()
            elif action_id == 7:
                self.handle_take_photo()
            elif action_id == 8:
                self.handle_image_analysis()
            
            if not self.debug_mode:
                self.tts_queue.put("指令已执行")
        except Exception as e:
            error_msg = f"执行出错：{str(e)}"
            self.tts_queue.put(error_msg)
            print(error_msg)

    def handle_take_photo(self):
        """处理拍照命令"""
        code, data = self.video_client.GetImageSample()
        if code == 0:
            self.last_photo_path = self.web_interface.save_photo(bytes(data))
            self.tts_queue.put(f"拍照成功，已保存为{self.last_photo_path}")
            # 拍照后直接进行识别
            self.handle_image_analysis()
        else:
            self.tts_queue.put("拍照失败，请重试")

    def handle_image_analysis(self):
        """优化后的图像分析处理"""
        if not self.last_photo_path:
            self.tts_queue.put("请先拍摄照片")
            return
        
        if not os.path.exists(self.last_photo_path):
            self.tts_queue.put("照片文件不存在")
            return
        
        items = self.detector.analyze(self.last_photo_path)
        if items:
            result = "、".join(items[:3])  # 最多返回3个结果
            self.tts_queue.put(f"识别到：{result}")
        else:
            self.tts_queue.put("未识别到有效物品")

    def analyze_image(self, image_path):
        """执行图像识别"""
        if not self.model:
            return "图像识别功能不可用"

        try:
            # 图像预处理
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 执行推理
            results = self.model([img], size=640)
            
            # 解析结果
            detections = results.pandas().xyxy[0]
            items = detections[detections['confidence'] > 0.5]['name'].unique()
            
            # 翻译字典
            label_translation = {
                'person': '人', 'car': '汽车', 'chair': '椅子',
                'book': '书', 'cell phone': '手机', 'cup': '杯子',
                'laptop': '笔记本电脑', 'dog': '狗', 'cat': '猫',
                'bottle': '瓶子', 'keyboard': '键盘', 'mouse': '鼠标',
                'tv': '电视', 'umbrella': '雨伞', 'backpack': '背包'
            }
            
            # 翻译和去重
            translated_items = []
            for item in items:
                translated = label_translation.get(item.lower(), item)
                if translated not in translated_items:
                    translated_items.append(translated)
            
            return '、'.join(translated_items) if translated_items else None
            
        except Exception as e:
            print(f"识别错误: {str(e)}")
            return None

    def cleanup_resources(self):
        """资源清理"""
        print("正在清理资源...")
        self.running = False
        
        # 音频资源
        if self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_interface.terminate()
        
        # 运动控制
        try:
            self.sport_client.StandDown()
            time.sleep(2)
            self.sport_client.StopMove()
        except:
            pass
        
        # 视频资源
        try:
            del self.video_client
        except:
            pass
            
        # 关闭ROS节点
        try:
            rospy.signal_shutdown("程序退出")
        except:
            pass
            
        # 释放检测器资源
        if self.detector and self.detector.model:
            del self.detector.model
            torch.cuda.empty_cache()
        
        # 事件循环
        if self.loop.is_running():
            self.loop.stop()
        
        # 清空队列
        while not self.tts_queue.empty():
            self.tts_queue.get_nowait()
        while not self.asr_queue.empty():
            self.asr_queue.get_nowait()


    def run(self):
        """主运行逻辑"""
        try:
            self.execute_action(1)  # 初始平衡站立
            self.tts_queue.put("系统已启动，准备就绪")  # 修改启动提示
            
            # 直接启动TTS线程
            self.tts_thread = threading.Thread(
                target=self.loop.run_forever,
                daemon=True
            )
            self.tts_thread.start()
    
            asyncio.run_coroutine_threadsafe(self.tts_worker(), self.loop)
    
            threads = [
                threading.Thread(target=self.network_monitor),
                #threading.Thread(target=self.online_asr_processing),
                threading.Thread(target=self.offline_asr_processing)
            ]
    
            for t in threads:
                t.daemon = True
                t.start()
    
            while self.running:
                try:
                    command = self.cmd_queue.get(timeout=0.5)
                    self.process_command(command)
                except queue.Empty:
                    continue
        finally:
            self.cleanup_resources()
            print("程序已安全退出")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    ChannelFactoryInitialize(0, sys.argv[1])
    
    try:
        controller = VoiceControl()
        controller.run()
    except Exception as e:
        print(f"系统错误: {str(e)}")
    finally:
        print("程序退出")
