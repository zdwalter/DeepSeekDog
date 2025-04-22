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
from collections import deque
from datetime import datetime
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

# ---------- 新增：SLAM 相关库 ----------
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
import pickle

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
MAPS_DIR = "static/maps"  # 地图保存目录
MAP_RESOLUTION = 0.05  # 地图分辨率 (米/像素)
MAP_SIZE = 200  # 地图尺寸 (像素)
MAX_EXPLORE_TIME = 300  # 最大探索时间 (秒)

# ============================================================
#  SLAM 建图类
# ============================================================
import heapq
import math
from collections import deque

class SLAMMapper:
    """SLAM 建图和导航类，带路径规划和避障功能"""
    
    def __init__(self):
        # 地图参数
        self.map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)  # 占据网格地图
        self.origin = np.array([MAP_SIZE//2, MAP_SIZE//2])  # 地图原点 (中心)
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta] (米, 米, 弧度)
        
        # 状态标志
        self.exploring = False
        self.navigating = False
        self.target_position = None
        
        # 线程控制
        self.explore_thread = None
        self.navigate_thread = None
        self.save_thread = None
        
        # 数据记录
        self.map_lock = threading.Lock()
        self.path_history = deque(maxlen=1000)  # 路径历史记录
        self.planned_path = []  # 当前规划路径
        
        # 避障参数
        self.obstacle_threshold = 0.7  # 占据概率阈值
        self.safety_distance = 0.3  # 安全距离(米)
        self.safety_pixels = int(self.safety_distance / MAP_RESOLUTION)
        
        # 创建目录
        os.makedirs(MAPS_DIR, exist_ok=True)
        os.makedirs("static/photos", exist_ok=True)
    
    # ---------- 基础地图操作 ----------
    def reset_map(self):
        """重置地图"""
        with self.map_lock:
            self.map.fill(0)
            self.robot_pose = np.array([0.0, 0.0, 0.0])
            self.path_history.clear()
            self.planned_path = []
    
    def world_to_map(self, world_coords):
        """世界坐标转地图坐标"""
        map_coords = (world_coords[:2] / MAP_RESOLUTION + self.origin).astype(int)
        return np.clip(map_coords, 0, MAP_SIZE-1)
    
    def map_to_world(self, map_coords):
        """地图坐标转世界坐标"""
        return (map_coords - self.origin) * MAP_RESOLUTION
    
    def update_map(self, point_cloud):
        """用点云数据更新地图"""
        if point_cloud is None:
            return
            
        try:
            # 获取有效点
            valid_points = []
            for p in point_cloud:
                if abs(p['x']) < 5.0 and abs(p['y']) < 5.0:  # 限制范围
                    valid_points.append([p['x'], p['y']])
            
            if not valid_points:
                return
                
            # 转换到地图坐标系
            points = np.array(valid_points)
            map_points = self.world_to_map(points)
            
            # 更新占据网格
            with self.map_lock:
                # 清除旧的占据信息
                self.map.fill(0)
                
                # 标记障碍物
                for x, y in map_points:
                    self.map[y, x] = 1.0  # 占据
                
                # 标记机器人当前位置
                robot_map_pos = self.world_to_map(self.robot_pose[:2])
                self.map[robot_map_pos[1], robot_map_pos[0]] = 0.5  # 机器人位置
                
                # 高斯平滑
                self.map = gaussian_filter(self.map, sigma=1.0)
                
        except Exception as e:
            print(f"地图更新错误: {e}")
    
    # ---------- 路径规划与避障 ----------
    def is_valid_position(self, map_pos):
        """检查地图位置是否有效(无碰撞)"""
        x, y = map_pos
        if x < 0 or x >= MAP_SIZE or y < 0 or y >= MAP_SIZE:
            return False
            
        # 检查安全区域
        x_min = max(0, x - self.safety_pixels)
        x_max = min(MAP_SIZE-1, x + self.safety_pixels)
        y_min = max(0, y - self.safety_pixels)
        y_max = min(MAP_SIZE-1, y + self.safety_pixels)
        
        # 检查安全区域内是否有障碍物
        return np.max(self.map[y_min:y_max+1, x_min:x_max+1]) < self.obstacle_threshold
    
    def heuristic(self, a, b):
        """A*启发式函数(欧几里得距离)"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def a_star_path_planning(self, start_pos, goal_pos):
        """A*路径规划算法"""
        start = tuple(self.world_to_map(start_pos))
        goal = tuple(self.world_to_map(goal_pos))
        
        # 优先队列 (f_score, node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # 路径记录
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # 邻居方向 (8连通)
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), 
                    (1,1), (1,-1), (-1,1), (-1,-1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return [self.map_to_world(np.array(p)) for p in path]
            
            for dx, dy in neighbors:
                neighbor = (current[0]+dx, current[1]+dy)
                
                if not self.is_valid_position(neighbor):
                    continue
                
                # 对角线移动成本更高
                tentative_g = g_score[current] + (1.4 if dx and dy else 1.0)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 没有找到路径
    
    def get_obstacle_direction(self):
        """检测障碍物方向(简单版本)"""
        robot_pos = self.world_to_map(self.robot_pose[:2])
        directions = {
            'front': (0, 1),
            'back': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        obstacle_dirs = []
        for name, (dx, dy) in directions.items():
            x, y = robot_pos[0] + dx*self.safety_pixels, robot_pos[1] + dy*self.safety_pixels
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and self.map[y, x] >= self.obstacle_threshold:
                obstacle_dirs.append(name)
        
        return obstacle_dirs
    
    # ---------- 导航控制 ----------
    def navigate_to(self, sport_client, target_pos):
        """导航到指定位置"""
        if self.navigating:
            return False
            
        self.target_position = target_pos
        self.navigating = True
        
        self.navigate_thread = threading.Thread(
            target=self._navigate_worker,
            args=(sport_client,),
            daemon=True
        )
        self.navigate_thread.start()
        return True
    
    def _navigate_worker(self, sport_client):
        """导航工作线程"""
        try:
            while self.navigating and self.target_position is not None:
                # 计算当前到目标的距离
                dx = self.target_position[0] - self.robot_pose[0]
                dy = self.target_position[1] - self.robot_pose[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                # 如果已经很接近目标，停止
                if distance < 0.3:  # 30cm阈值
                    sport_client.StopMove()
                    self.navigating = False
                    break
                
                # 规划路径
                path = self.a_star_path_planning(self.robot_pose[:2], self.target_position)
                if not path:
                    print("无法找到路径到目标位置")
                    self.navigating = False
                    break
                
                self.planned_path = path
                
                # 获取下一个路径点(向前看1米)
                lookahead_dist = 1.0  # 米
                next_point = None
                for point in path:
                    dist = math.sqrt((point[0]-self.robot_pose[0])**2 + 
                                   (point[1]-self.robot_pose[1])**2)
                    if dist >= lookahead_dist:
                        next_point = point
                        break
                
                if next_point is None:
                    next_point = path[-1]  # 如果不够远，直接去最后一个点
                
                # 计算需要转向的角度
                target_angle = math.atan2(next_point[1]-self.robot_pose[1], 
                                         next_point[0]-self.robot_pose[0])
                angle_diff = (target_angle - self.robot_pose[2] + math.pi) % (2*math.pi) - math.pi
                
                # 控制命令
                speed = min(0.5, distance)  # 最大速度0.5m/s
                turn_rate = np.clip(angle_diff, -0.5, 0.5)
                
                # 避障检查
                obstacle_dirs = self.get_obstacle_direction()
                if 'front' in obstacle_dirs and distance < 1.0:
                    # 前方有障碍且距离较近，减速
                    speed *= 0.5
                    if abs(angle_diff) > math.pi/4:  # 角度偏差大时优先转向
                        speed = 0.1
                
                sport_client.Move(speed, 0, turn_rate)
                time.sleep(0.1)
                
        finally:
            sport_client.StopMove()
            self.navigating = False
    
    # ---------- 探索控制 ----------
    def explore_environment(self, sport_client, duration=MAX_EXPLORE_TIME):
        """自动探索环境(带避障)"""
        if self.exploring:
            return False
            
        self.exploring = True
        
        # 启动探索线程
        self.explore_thread = threading.Thread(
            target=self._explore_worker, 
            args=(sport_client, duration),
            daemon=True
        )
        self.explore_thread.start()
        
        # 启动自动保存地图线程
        self.save_thread = threading.Thread(
            target=self._auto_save_map_worker,
            daemon=True
        )
        self.save_thread.start()
        
        return True
    
    def _explore_worker(self, sport_client, duration):
        """探索工作线程(改进版带避障)"""
        start_time = time.time()
        last_turn_time = time.time()
        turn_direction = 1  # 1 for right, -1 for left
        
        try:
            while self.exploring and (time.time() - start_time) < duration:
                # 检查前方障碍物
                obstacle_dirs = self.get_obstacle_direction()
                if 'front' in obstacle_dirs:
                    # 前方有障碍物，后退并转向
                    sport_client.Move(-0.2, 0, 0)
                    time.sleep(0.5)
                    sport_client.Move(0, 0, 0.5 * turn_direction)
                    time.sleep(1.0)
                    turn_direction *= -1  # 下次转向相反方向
                    last_turn_time = time.time()
                else:
                    # 正常前进
                    sport_client.Move(0.3, 0, 0)
                
                # 每隔几秒随机转向
                if time.time() - last_turn_time > 5.0:
                    turn_direction *= -1
                    sport_client.Move(0, 0, 0.3 * turn_direction)
                    last_turn_time = time.time()
                    time.sleep(1.0)
                
                # 更新位姿 (简化版，实际应该用里程计)
                self.update_pose(0.05, 0, 0.01 * turn_direction)
                time.sleep(0.1)
                
        finally:
            sport_client.StopMove()
            self.exploring = False
    
    def stop_exploration(self):
        """停止探索"""
        self.exploring = False
        if self.explore_thread:
            self.explore_thread.join(timeout=1.0)
        if hasattr(self, 'save_thread') and self.save_thread:
            self.save_thread.join(timeout=1.0)
    
    # ---------- 地图持久化 ----------
    def save_map(self, filename=None):
        """保存当前地图到文件"""
        if filename is None:
            filename = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
        filepath = os.path.join(MAPS_DIR, filename)
        
        with self.map_lock:
            map_data = {
                'map': self.map.copy(),
                'resolution': MAP_RESOLUTION,
                'origin': self.origin,
                'robot_pose': self.robot_pose,
                'path_history': list(self.path_history)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(map_data, f)
                
            # 同时保存地图图片
            self._save_map_image()
                
        print(f"地图已保存到 {filepath}")
        return filepath
    
    def _save_map_image(self):
        """内部方法：保存地图图片"""
        map_img = self.get_map_image()
        if map_img is not None:
            img_path = os.path.join("static/photos", "slam_map.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(map_img, cv2.COLOR_RGBA2BGR))
    
    def load_map(self, filename):
        """从文件加载地图"""
        filepath = os.path.join(MAPS_DIR, filename)
        
        try:
            with open(filepath, 'rb') as f:
                map_data = pickle.load(f)
                
            with self.map_lock:
                self.map = map_data['map']
                self.origin = map_data['origin']
                self.robot_pose = map_data['robot_pose']
                self.path_history = deque(map_data['path_history'], maxlen=1000)
                
            # 加载后立即保存图片
            self._save_map_image()
                
            print(f"已加载地图 {filename}")
            return True
            
        except Exception as e:
            print(f"加载地图失败: {e}")
            return False
    
    def get_map_image(self):
        """获取当前地图的可视化图像"""
        with self.map_lock:
            # 创建彩色地图
            cmap = plt.cm.get_cmap('viridis')
            colored_map = cmap(self.map)
            
            # 添加路径历史
            if len(self.path_history) > 1:
                path = np.array(self.path_history)
                path_map = self.world_to_map(path[:, :2])
                for x, y in path_map:
                    colored_map[y, x] = [1.0, 0.0, 0.0, 1.0]  # 红色路径
            
            # 添加规划路径
            if self.planned_path:
                path = np.array(self.planned_path)
                path_map = self.world_to_map(path[:, :2])
                for x, y in path_map:
                    colored_map[y, x] = [0.0, 1.0, 0.0, 1.0]  # 绿色规划路径
                    
            # 添加机器人当前位置
            robot_pos = self.world_to_map(self.robot_pose[:2])
            cv2.circle(colored_map, (robot_pos[0], robot_pos[1]), 3, (0, 0, 1.0), -1)
            
            # 添加目标位置
            if self.target_position is not None:
                target_pos = self.world_to_map(self.target_position[:2])
                cv2.circle(colored_map, (target_pos[0], target_pos[1]), 5, (1.0, 0.0, 1.0), -1)
            
            # 旋转地图使机器人始终朝上
            angle = np.degrees(self.robot_pose[2])
            rotated_map = rotate(colored_map, angle, center=(robot_pos[0], robot_pos[1]), order=0)
            
            return (rotated_map * 255).astype(np.uint8)
    
    def update_pose(self, dx, dy, dtheta):
        """更新机器人位姿"""
        with self.map_lock:
            # 更新位置
            self.robot_pose[0] += dx
            self.robot_pose[1] += dy
            self.robot_pose[2] = (self.robot_pose[2] + dtheta) % (2 * np.pi)
            
            # 记录路径
            self.path_history.append(self.robot_pose.copy())
    
    def _auto_save_map_worker(self):
        """自动保存地图图片的工作线程"""
        while self.exploring or self.navigating:
            try:
                # 保存当前地图状态
                self._save_map_image()
                time.sleep(1.0)  # 每秒保存一次
            except Exception as e:
                print(f"自动保存地图错误: {e}")
                time.sleep(1.0)


# ============================================================
#  Lidar 处理类 (更新版)
# ============================================================
class LidarProcessor:
    """LIDAR 数据处理"""
    
    def __init__(self, slam_mapper=None):
        self.latest_cloud_data = None
        self.lidar_running = False
        self.lidar_thread = None
        self.cloud_subscriber = None
        self.obstacle_distance = None
        self.scan_complete = False
        self.slam_mapper = slam_mapper  # SLAM 建图器
        self.point_cloud_cache = []
        
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
        
    def point_cloud_callback(self, cloud_data: PointCloud2_):
        if cloud_data is not None:
            self.latest_cloud_data = cloud_data
            point_cloud = self._process_point_cloud(cloud_data)
            
            # 更新SLAM地图
            if self.slam_mapper and point_cloud:
                self.slam_mapper.update_map(point_cloud)
                
            # 障碍物检测
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
                bbox_to_anchor=(1.25, 1.0))
            
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
#  Web 接口 (更新版)
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
        
        # 新增地图路由
        self.app.add_url_rule("/maps", "get_maps", self.get_maps, methods=["GET"])
        self.app.add_url_rule("/map/<filename>", "get_map", self.get_map)
        self.app.add_url_rule("/slam_map", "get_slam_map", self.get_slam_map)

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

    def get_maps(self):
        """获取所有保存的地图列表"""
        maps = []
        if os.path.exists(MAPS_DIR):
            for f in os.listdir(MAPS_DIR):
                if f.endswith('.pkl'):
                    maps.append({
                        "filename": f,
                        "timestamp": os.path.getmtime(os.path.join(MAPS_DIR, f)),
                        "url": f"/map/{f}"
                    })
        return jsonify(sorted(maps, key=lambda x: x["timestamp"], reverse=True))
    
    def get_map(self, filename):
        """获取地图文件"""
        return send_from_directory(MAPS_DIR, filename)
    
    def get_slam_map(self):
        """获取当前SLAM地图图像"""
        if hasattr(self.voice_controller, "slam_mapper"):
            map_image = self.voice_controller.slam_mapper.get_map_image()
            if map_image is not None:
                # 保存为临时图片
                map_path = "static/photos/slam_map.jpg"
                cv2.imwrite(map_path, cv2.cvtColor(map_image, cv2.COLOR_RGBA2BGR))
                
                return jsonify({
                    "url": "/photo/slam_map.jpg",
                    "timestamp": time.time()
                })
        return jsonify({"error": "No map available"}), 404
        
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

        # 新增SLAM建图器
        self.slam_mapper = SLAMMapper()
        
        # 更新Lidar处理器
        self.lidar_processor = LidarProcessor(self.slam_mapper)
        self.lidar_processor.start_lidar()
        
        # 新增命令映射
        self.command_map.update({
            r"开始建图$": 20,
            r"停止建图$": 21,
            r"保存地图$": 22,
            r"加载地图$": 23,
            r"查看地图$": 24,
            r"start mapping$": 20,
            r"stop mapping$": 21,
            r"save map$": 22,
            r"load map$": 23,
            r"show map$": 24,
        })
        
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
            # 新增SLAM相关动作
            elif action_id == 20:  # 开始建图
                if self.slam_mapper.explore_environment(self.sport_client):
                    self.tts_queue.put("开始建图，机器人将自动探索环境")
                else:
                    self.tts_queue.put("建图已在运行中")
                    
            elif action_id == 21:  # 停止建图
                self.slam_mapper.stop_exploration()
                self.tts_queue.put("已停止建图")
                
            elif action_id == 22:  # 保存地图
                map_path = self.slam_mapper.save_map()
                self.tts_queue.put(f"地图已保存到 {os.path.basename(map_path)}")
                
            elif action_id == 23:  # 加载地图
                # 实际应用中应该让用户选择地图文件
                maps = [f for f in os.listdir(MAPS_DIR) if f.endswith('.pkl')]
                if maps:
                    latest_map = max(maps, key=lambda f: os.path.getmtime(os.path.join(MAPS_DIR, f)))
                    if self.slam_mapper.load_map(latest_map):
                        self.tts_queue.put(f"已加载地图 {latest_map}")
                    else:
                        self.tts_queue.put("加载地图失败")
                else:
                    self.tts_queue.put("没有找到地图文件")
                    
            elif action_id == 24:  # 查看地图
                self.tts_queue.put("正在显示当前地图")
                # 前端将通过WebSocket请求地图图像
        
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
        
        # 停止SLAM建图
        if hasattr(self, "slam_mapper"):
            self.slam_mapper.stop_exploration()
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
