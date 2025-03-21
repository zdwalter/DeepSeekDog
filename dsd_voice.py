import time
import sys
import threading
import queue
import socket
import json
import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

class VoiceControl:
    def __init__(self):
        # 多线程通信队列
        self.cmd_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # 启动确认相关
        self.startup_confirmed = False
        self.startup_timer = None
        
        # 网络状态检测
        self.online_mode = self.check_network()
        
        # 语音模块初始化
        self.init_voice_engine()
        
        # 运动控制初始化
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # 控制参数
        self.running = True
        self.debug_mode = False
        self.command_map = {
            "确认模式": -3,
            "停": 6, "stop": 6, "暂停": 6,
            "前进": 3, "forward": 3,
            "后退": 3, "back": 3,
            "左移": 4, "left": 4,
            "右移": 4, "right": 4,
            "旋转": 5, "turn": 5,
            "站立": 1, "stand": 1,
            "坐下": 2, "sit": 2,
            "平衡站": 9, "balance": 9,
            "调试模式": -1,
            "正常模式": -2,
            "退出": 0
        }

    def init_voice_engine(self):
        """初始化语音引擎"""
        # 离线TTS引擎
        self.tts_engine = pyttsx3.init()
        
        # 离线ASR模型（需要提前下载模型文件）
        self.vosk_model = Model(lang="cn")
        
        # 在线识别引擎
        self.online_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # 音频输入流
        self.audio_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )

    def check_network(self):
        """检测网络连接状态（使用Bing测试）"""
        try:
            socket.create_connection(("www.bing.com", 80), timeout=3)
            return True
        except OSError:
            return False

    def tts_thread(self):
        """语音合成线程"""
        while self.running:
            try:
                text = self.audio_queue.get(timeout=1)
                if text:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except queue.Empty:
                continue

    def online_asr_thread(self):
        """在线语音识别线程"""
        with self.microphone as source:
            self.online_recognizer.adjust_for_ambient_noise(source)
            while self.running:
                try:
                    audio = self.online_recognizer.listen(source, timeout=5)
                    text = self.online_recognizer.recognize_google(audio, language='zh-CN')
                    self.cmd_queue.put(text)
                except (sr.UnknownValueError, sr.WaitTimeoutError):
                    continue
                except sr.RequestError:
                    self.online_mode = False

    def offline_asr_thread(self):
        """离线语音识别线程"""
        rec = KaldiRecognizer(self.vosk_model, 16000)
        while self.running:
            data = self.audio_stream.read(4000)
            if rec.AcceptWaveform(data):
                result = rec.Result()
                try:
                    text = json.loads(result)["text"]
                    if text:
                        self.cmd_queue.put(text)
                except json.JSONDecodeError:
                    continue

    def network_monitor_thread(self):
        """网络状态监控线程"""
        while self.running:
            current_status = self.check_network()
            if current_status != self.online_mode:
                self.online_mode = current_status
                status = "在线" if self.online_mode else "离线"
                self.audio_queue.put(f"网络状态已切换至{status}模式")
            time.sleep(10)

    def startup_timeout(self):
        """启动确认超时处理"""
        if not self.startup_confirmed:
            self.audio_queue.put("启动确认超时，程序即将退出")
            time.sleep(2)
            self.running = False

    def process_command(self, command):
        """处理语音命令"""
        cmd = command.lower()
        
        # 启动确认阶段处理
        if not self.startup_confirmed:
            if "确认模式" in cmd:
                self.startup_confirmed = True
                if self.startup_timer:
                    self.startup_timer.cancel()
                self.audio_queue.put("启动确认成功，进入操作模式")
                return
            else:
                return

        # 模式切换命令
        if "调试" in cmd:
            self.debug_mode = True
            self.audio_queue.put("已进入调试模式")
            return
        elif "正常" in cmd:
            self.debug_mode = False 
            self.audio_queue.put("已进入正常模式")
            return
        elif "退出" in cmd:
            self.running = False
            self.audio_queue.put("程序结束")
            return

        # 运动控制命令处理
        action_id = None
        for key in self.command_map:
            if key in cmd:
                action_id = self.command_map[key]
                break

        # 执行命令逻辑
        if action_id is not None:
            self.execute_action(action_id, cmd)
        else:
            self.audio_queue.put("指令未识别")

    def execute_action(self, action_id, raw_cmd):
        """执行具体动作"""
        try:
            if action_id == 6:
                self.sport_client.StopMove()
            elif action_id == 3:
                if "back" in raw_cmd or "后退" in raw_cmd:
                    self.sport_client.Move(-0.3, 0, 0)
                else:
                    self.sport_client.Move(0.3, 0, 0)
            elif action_id == 4:
                if "right" in raw_cmd or "右移" in raw_cmd:
                    self.sport_client.Move(0, -0.3, 0)
                else:
                    self.sport_client.Move(0, 0.3, 0)
            elif action_id == 5:
                self.sport_client.Move(0, 0, 0.5)
            elif action_id == 1:
                self.sport_client.StandUp()
            elif action_id == 2:
                self.sport_client.StandDown()
            elif action_id == 9:
                self.sport_client.BalanceStand()
            
            if not self.debug_mode:
                self.audio_queue.put("指令已执行")
                
        except Exception as e:
            self.audio_queue.put(f"执行出错：{str(e)}")

    def run(self):
        """主运行逻辑"""
        # 启动确认阶段
        self.audio_queue.put("请说'确认模式'以启动程序，您有10秒时间")
        self.startup_timer = threading.Timer(10.0, self.startup_timeout)
        self.startup_timer.start()

        # 启动线程
        threads = [
            threading.Thread(target=self.tts_thread),
            threading.Thread(target=self.network_monitor_thread),
        ]
        
        # 根据模式启动识别线程
        if self.online_mode:
            threads.append(threading.Thread(target=self.online_asr_thread))
        else:
            threads.append(threading.Thread(target=self.offline_asr_thread))

        for t in threads:
            t.daemon = True
            t.start()

        # 主处理循环
        try:
            while self.running:
                try:
                    command = self.cmd_queue.get(timeout=1)
                    if command:
                        self.process_command(command)
                except queue.Empty:
                    if not self.startup_confirmed and not self.startup_timer.is_alive():
                        self.running = False
        finally:
            self.sport_client.StopMove()
            if self.startup_timer:
                self.startup_timer.cancel()
            self.audio_queue.put("系统已关闭")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    ChannelFactoryInitialize(0, sys.argv[1])
    
    try:
        controller = VoiceControl()
        controller.run()
    except KeyboardInterrupt:
        print("程序被手动终止")
