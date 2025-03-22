import time
import sys
import threading
import queue
import socket
import json
import re
import signal
import speech_recognition as sr
import edge_tts
import asyncio
import subprocess
from vosk import Model, KaldiRecognizer
import pyaudio
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

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
        self.startup_confirmed = False
        self.startup_timer = None
        self.debug_mode = True

        # 网络状态
        self.current_network_status = self.check_network()
        
        # 音频设备初始化
        self.audio_interface = pyaudio.PyAudio()
        self.init_voice_engine()

        # 运动控制初始化
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()


        
        # 命令映射表
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

    async def tts_worker(self):
        """TTS工作协程"""
        while self.running:
            try:
                text = self.tts_queue.get_nowait()
                if text:
                    print(f"[TTS] {text}")
                    communicate = edge_tts.Communicate(
                        text=text,
                        voice="zh-CN-YunxiNeural",  # 使用中文语音
                        rate="+10%", 
                        volume="+0%"
                    )
                    
                    # 使用FFmpeg实时播放
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
            # 离线ASR模型
            self.vosk_model = Model(lang="cn")
            
            # 在线识别设备
            self.online_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # 音频输入流
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
        return False #debug
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
        # 启动确认处理
        if not self.startup_confirmed:
            if re.search(r"确认模式$", cmd):
                self.startup_confirmed = True
                if self.startup_timer:
                    self.startup_timer.cancel()
                self.tts_queue.put("启动确认成功，进入操作模式")
            return

        # 模式切换命令
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

        # 运动控制命令
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

    def execute_action(self, action_id, raw_cmd):
        """执行动作"""
        print(f"执行动作{raw_cmd}, {action_id}")
        step = 0.3
        try:
            if action_id == 6:
                self.sport_client.StopMove()
            elif action_id == 3:
                speed = -step if any(x in raw_cmd for x in ["后", "back"]) else step
                self.sport_client.Move(speed, 0, 0)
            elif action_id == 4:
                speed = -step if any(x in raw_cmd for x in ["右", "right"]) else step
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
                self.tts_queue.put("指令已执行")
        except Exception as e:
            error_msg = f"执行出错：{str(e)}"
            self.tts_queue.put(error_msg)
            print(error_msg)
            
    def cleanup_resources(self):
        """资源清理"""
        print("正在清理资源...")
        self.running = False
        
        # 停止音频流
        if self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_interface.terminate()
        
        # 停止运动
        try:
            self.sport_client.StandDown()
            time.sleep(2)
            self.sport_client.StopMove()
        except:
            pass
        
        # 停止事件循环
        if self.loop.is_running():
            self.loop.stop()
        
        # 清空队列
        while not self.tts_queue.empty():
            self.tts_queue.get_nowait()
        while not self.asr_queue.empty():
            self.asr_queue.get_nowait()

    def startup_timeout(self):
        """启动超时处理"""
        if not self.startup_confirmed:
            print("启动确认超时")
            self.tts_queue.put("启动确认超时，程序即将退出")
            time.sleep(2)
            self.running = False

    def run(self):
        """主运行逻辑"""
        try:

            if 1:
                #self.sport_client.StandDown()
                #time.sleep(2)
                #self.sport_client.StandUp()
                #time.sleep(2)
                self.tts_queue.put("启动测试：前进")
                self.sport_client.Move(0.3, 0, 0)
                time.sleep(2)
                self.tts_queue.put("启动测试：后退")
                self.sport_client.Move(-0.3, 0, 0)
            # 启动确认
            self.tts_queue.put("请说'确认模式'以启动程序，您有10秒时间")
            self.startup_timer = threading.Timer(10.0, self.startup_timeout)
            self.startup_timer.start()

            # 启动TTS事件循环
            self.tts_thread = threading.Thread(
                target=self.loop.run_forever,
                daemon=True
            )
            self.tts_thread.start()

            # 将TTS worker加入事件循环
            asyncio.run_coroutine_threadsafe(self.tts_worker(), self.loop)

            # 启动其他线程
            threads = [
                threading.Thread(target=self.network_monitor),
                threading.Thread(target=self.online_asr_processing),
                threading.Thread(target=self.offline_asr_processing)
            ]

            for t in threads:
                t.daemon = True
                t.start()


            
            # 主循环
            while self.running:
                try:
                    command = self.cmd_queue.get(timeout=0.5)
                    self.process_command(command)
                except queue.Empty:
                    continue
        finally:
            if self.startup_timer and self.startup_timer.is_alive():
                self.startup_timer.cancel()
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
