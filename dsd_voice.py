import time
import sys
import threading
import queue
import socket
import json
import re
import signal
import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

class VoiceControl:
    def __init__(self):
        # 初始化信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 初始化TTS引擎
        self.tts_engine = pyttsx3.init()
        self.tts_lock = threading.Lock()  # 新增线程锁
        self.active_utterance = None      # 跟踪当前语音片段

        # 设置引擎回调
        self.tts_engine.connect('started-utterance', self.on_start)
        self.tts_engine.connect('finished-utterance', self.on_end)
        
        # 多线程通信队列
        self.cmd_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # 状态控制
        self.running = True
        self.startup_confirmed = False
        self.startup_timer = None
        self.debug_mode = False
        
        # 网络状态
        self.current_network_status = self.check_network()
        
        # 音频资源初始化
        self.audio_interface = pyaudio.PyAudio()
        self.init_voice_engine()
        
        # 运动控制初始化
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        # 命令映射表（正则表达式优化）
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

    def signal_handler(self, signum, frame):
        """信号处理函数"""
        self.running = False
        print("\n接收到终止信号，正在清理资源...")

    def init_voice_engine(self):
        """增强型语音引擎初始化"""
        try:
            # TTS引擎
            self.tts_engine = pyttsx3.init()
            
            # 离线ASR模型
            self.vosk_model = Model(lang="cn")
            
            # 初始化在线识别设备（新增麦克风初始化）
            self.online_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()  # 添加缺失的麦克风初始化
            
            # 音频输入流（增加异常处理）
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
            raise RuntimeError(f"语音引擎初始化失败: {str(e)}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """增强型音频回调"""
        if self.running:
            self.audio_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
        else:
            return (None, pyaudio.paComplete)

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

    def on_start(self, name):
        """语音开始回调"""
        with self.tts_lock:
            self.active_utterance = name
            
    def on_end(self, name, completed):
        """语音结束回调"""
        with self.tts_lock:
            self.active_utterance = None
            
    def tts_thread(self):
        """增强型TTS线程"""
        while self.running:
            try:
                text = self.audio_queue.get(timeout=0.3)  # 更短的超时时间
                if text and self.running:
                    # 生成唯一ID用于跟踪
                    utterance_id = str(hash(text + str(time.time())))
                    
                    with self.tts_lock:
                        self.tts_engine.say(text, utterance_id)
                        # 启动异步事件循环
                        self.tts_engine.startLoop(False)
                        
                        # 保持循环直到语音完成或停止信号
                        while self.running and self.active_utterance == utterance_id:
                            self.tts_engine.iterate()
                            time.sleep(0.1)
                            
                        self.tts_engine.endLoop()

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
        """增强型资源清理（带调试日志）"""
        print("[DEBUG] 开始清理资源...")
        
        # 设置运行状态为False
        print("[DEBUG] 设置running=False")
        self.running = False
        
        # 清空音频队列防止阻塞
        print("[DEBUG] 清空音频队列")
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        # 停止音频流（分步骤记录）
        try:
            print("[DEBUG] 正在停止音频流...")
            if self.audio_stream.is_active():
                self.audio_stream.stop_stream()
                print("[DEBUG] 音频流已停止")
            else:
                print("[DEBUG] 音频流未处于活动状态")
        except Exception as e:
            print(f"[ERROR] 停止音频流时出错: {str(e)}")
        
        # 关闭音频流
        try:
            print("[DEBUG] 正在关闭音频流...")
            self.audio_stream.close()
            print("[DEBUG] 音频流已关闭")
        except Exception as e:
            print(f"[ERROR] 关闭音频流时出错: {str(e)}")
        
        # 终止PyAudio实例
        try:
            print("[DEBUG] 正在终止PyAudio...")
            self.audio_interface.terminate()
            print("[DEBUG] PyAudio已终止")
        except Exception as e:
            print(f"[ERROR] 终止PyAudio时出错: {str(e)}")
        
        # 停止TTS引擎
        """增强型资源清理"""
        print("[DEBUG] 正在强制停止TTS引擎...")
        try:
            # 停止当前语音播放
            with self.tts_lock:
                if self.active_utterance:
                    self.tts_engine.stop()
                    # 等待最多1秒
                    start = time.time()
                    while self.active_utterance and (time.time() - start) < 1:
                        time.sleep(0.1)
                    
            # 彻底停止事件循环
            if self.tts_engine._loop.is_alive():
                self.tts_engine.endLoop()
                
            print("[DEBUG] TTS引擎已强制停止")
        except Exception as e:
            print(f"[ERROR] 停止TTS引擎时出错: {str(e)}")
        
        # 停止运动控制
        try:
            print("[DEBUG] 发送最终停止指令...")
            self.sport_client.StopMove()
            print("[DEBUG] 运动控制已停止")
        except Exception as e:
            print(f"[ERROR] 停止运动控制时出错: {str(e)}")
        
        # 额外安全措施：强制终止可能卡住的线程
        print("[DEBUG] 检查线程状态：")
        for t in threading.enumerate():
            if t != threading.main_thread():
                print(f"[THREAD] {t.name} (alive={t.is_alive()})")
        
        print("[DEBUG] 资源清理完成")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """带状态检查的音频回调"""
        if self.running:
            try:
                self.audio_queue.put(in_data, timeout=0.1)
                return (in_data, pyaudio.paContinue)
            except queue.Full:
                print("[WARNING] 音频队列已满，丢弃数据")
                return (None, pyaudio.paContinue)
        else:
            print("[DEBUG] 音频回调已停止")
            return (None, pyaudio.paComplete)

    def run(self):
        """增强型主运行逻辑"""
        try:
            # 启动确认阶段
            self.audio_queue.put("请说'确认模式'以启动程序，您有10秒时间")
            self.startup_timer = threading.Timer(10.0, self.startup_timeout)
            self.startup_timer.start()

            # 启动工作线程
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
                    command = self.cmd_queue.get(timeout=0.5)
                    if command:
                        self.process_command(command)
                except queue.Empty:
                    if not self.startup_confirmed and not self.startup_timer.is_alive():
                        self.running = False
        finally:
            start_cleanup_time = time.time()
            print(f"[TIMING] 开始清理时间戳：{start_cleanup_time}")
            self.cleanup_resources()
            
            # 添加清理超时监控
            timeout = 5  # 最多等待5秒
            while threading.active_count() > 1:
                if time.time() - start_cleanup_time > timeout:
                    print("[WARNING] 清理超时，强制退出！")
                    os._exit(1)
                time.sleep(0.1)
            
            print(f"[TIMING] 总清理耗时：{time.time()-start_cleanup_time:.2f}秒")

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
        print("程序已安全退出")
