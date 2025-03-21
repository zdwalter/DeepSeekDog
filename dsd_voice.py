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
        
        # 语音模块初始化
        self.audio_interface = pyaudio.PyAudio()
        self.init_voice_engine()
        
        # 运动控制初始化
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # 控制参数
        self.running = True
        self.debug_mode = False
        self.current_network_status = self.check_network()
        self.command_map = {
            r"确认模式$": -3,
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
            r"退出$": 0
        }

    def init_voice_engine(self):
        """初始化语音引擎"""
        try:
            # TTS引擎
            self.tts_engine = pyttsx3.init()
            
            # 离线ASR模型
            self.vosk_model = Model(lang="cn")
            
            # 在线识别引擎
            self.online_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # 音频输入流
            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000,
                stream_callback=self.audio_callback
            )
        except Exception as e:
            self.cleanup_resources()
            raise RuntimeError(f"语音引擎初始化失败: {str(e)}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频采集回调函数"""
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def check_network(self):
        """增强型网络检测"""
        test_servers = [
            ("www.google.com", 80),
            ("www.baidu.com", 80),
            ("8.8.8.8", 53)
        ]
        for server in test_servers:
            try:
                socket.create_connection(server, timeout=2)
                return True
            except OSError:
                continue
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
            except Exception as e:
                print(f"TTS错误: {str(e)}")

    def online_asr_processing(self):
        """在线语音识别处理"""
        with self.microphone as source:
            self.online_recognizer.adjust_for_ambient_noise(source)
            while self.running and self.current_network_status:
                try:
                    audio = self.online_recognizer.listen(source, timeout=3)
                    text = self.online_recognizer.recognize_google(audio, language='zh-CN')
                    self.cmd_queue.put(text)
                except (sr.UnknownValueError, sr.WaitTimeoutError):
                    continue
                except sr.RequestError:
                    self.current_network_status = False
                    self.audio_queue.put("网络连接中断，切换至离线模式")

    def offline_asr_processing(self):
        """离线语音识别处理"""
        rec = KaldiRecognizer(self.vosk_model, 16000)
        while self.running and not self.current_network_status:
            data = self.audio_queue.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                try:
                    text = json.loads(result)["text"]
                    if text:
                        self.cmd_queue.put(text)
                except json.JSONDecodeError:
                    continue

    def network_monitor(self):
        """网络状态监控"""
        while self.running:
            new_status = self.check_network()
            if new_status != self.current_network_status:
                self.current_network_status = new_status
                status = "在线" if new_status else "离线"
                self.audio_queue.put(f"网络状态已切换至{status}模式")
            time.sleep(5)

    def startup_timeout(self):
        """启动确认超时处理"""
        if not self.startup_confirmed:
            self.audio_queue.put("启动确认超时，程序即将退出")
            time.sleep(2)
            self.running = False

    def process_command(self, command):
        """使用正则表达式精确匹配命令"""
        import re
        cmd = command.lower().replace(" ", "")
        
        if not self.startup_confirmed:
            if re.search(r"确认模式$", cmd):
                self.startup_confirmed = True
                if self.startup_timer:
                    self.startup_timer.cancel()
                self.audio_queue.put("启动确认成功，进入操作模式")
            return

        # 模式切换命令
        if re.search(r"调试模式$", cmd):
            self.debug_mode = True
            self.audio_queue.put("已进入调试模式")
            return
        elif re.search(r"正常模式$", cmd):
            self.debug_mode = False
            self.audio_queue.put("已进入正常模式")
            return
        elif re.search(r"退出$", cmd):
            self.running = False
            self.audio_queue.put("程序结束")
            return

        # 运动控制命令匹配
        action_id = None
        for pattern, cmd_id in self.command_map.items():
            if re.search(pattern, cmd):
                action_id = cmd_id
                break

        if action_id is not None:
            self.execute_action(action_id, cmd)
        else:
            self.audio_queue.put("指令未识别")

    def execute_action(self, action_id, raw_cmd):
        """增强型动作执行"""
        try:
            # 运动控制指令
            if action_id == 6:
                self.sport_client.StopMove()
            elif action_id == 3:
                speed = -0.3 if any(x in raw_cmd for x in ["后", "back"]) else 0.3
                self.sport_client.Move(speed, 0, 0)
            elif action_id == 4:
                speed = -0.3 if any(x in raw_cmd for x in ["右", "right"]) else 0.3
                self.sport_client.Move(0, speed, 0)
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
            error_msg = f"执行出错：{str(e)}"
            self.audio_queue.put(error_msg)
            print(error_msg)

    def cleanup_resources(self):
        """资源清理"""
        if self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_interface.terminate()
        self.tts_engine.stop()

    def run(self):
        """主运行逻辑"""
        try:
            # 启动确认阶段
            self.audio_queue.put("请说'确认模式'以启动程序，您有10秒时间")
            self.startup_timer = threading.Timer(10.0, self.startup_timeout)
            self.startup_timer.start()

            # 启动线程
            threads = [
                threading.Thread(target=self.tts_thread),
                threading.Thread(target=self.network_monitor),
                threading.Thread(target=self.online_asr_processing),
                threading.Thread(target=self.offline_asr_processing)
            ]

            for t in threads:
                t.daemon = True
                t.start()

            # 主处理循环
            while self.running:
                try:
                    command = self.cmd_queue.get(timeout=1)
                    if command:
                        self.process_command(command)
                except queue.Empty:
                    if not self.startup_confirmed and not self.startup_timer.is_alive():
                        self.running = False
        finally:
            self.cleanup_resources()
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
    except Exception as e:
        print(f"系统错误: {str(e)}")
    except KeyboardInterrupt:
        print("程序被手动终止")
