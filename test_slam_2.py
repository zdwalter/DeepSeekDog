#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器狗主控制程序 - 重构版

主要功能模块:
1. 语音控制 (ASR/TTS)
2. 运动控制 (Unitree SDK)
3. 视觉处理 (YOLOv5)
4. SLAM建图与导航
5. Web界面 (Flask/SocketIO)
"""

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
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
import pickle
from vosk import KaldiRecognizer, Model
import math
import heapq
import matplotlib.pyplot as plt

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

# 常量定义
TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"
MAPS_DIR = "static/maps"
MAP_RESOLUTION = 0.05  # 米/像素
MAP_SIZE = 200  # 像素
MAX_EXPLORE_TIME = 300  # 秒

# ============================================================
#  SLAM 建图与导航类
# ============================================================
class SLAMMapper:
    """增强SLAM 建图和导航系统 - 使用概率栅格建图和ICP匹配"""
    
    def __init__(self):
        # 概率栅格地图 (log-odds表示)
        self.map_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        self.map_probabilities = np.full((MAP_SIZE, MAP_SIZE), 0.5, dtype=np.float32)
        self.origin = np.array([MAP_SIZE//2, MAP_SIZE//2])
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.last_pose = self.robot_pose.copy()
        
        # 传感器模型参数
        self.l_occ = 0.85  # 占用概率
        self.l_free = 0.15  # 空闲概率
        self.l_prior = 0.5  # 先验概率
        self.log_occ = np.log(self.l_occ / (1 - self.l_occ))
        self.log_free = np.log(self.l_free / (1 - self.l_free))
        
        self.exploring = False
        self.navigating = False
        self.target_position = None
        
        self.map_lock = threading.Lock()
        self.path_history = deque(maxlen=1000)
        self.planned_path = []
        
        # 改进的参数
        self.obstacle_threshold = 0.7
        self.safety_distance = 0.3  # 米
        self.safety_pixels = int(self.safety_distance / MAP_RESOLUTION)
        
        # ICP匹配相关
        self.last_scan_points = None
        self.icp_max_correspondence_distance = 0.1
        self.icp_max_iterations = 50
        
        # 探索相关
        self.frontiers = []
        self.exploration_targets = []
        
        os.makedirs(MAPS_DIR, exist_ok=True)
        os.makedirs("static/photos", exist_ok=True)
    
    def reset_map(self):
        """重置地图"""
        with self.map_lock:
            self.map_log_odds.fill(0)
            self.map_probabilities.fill(0.5)
            self.robot_pose = np.array([0.0, 0.0, 0.0])
            self.last_pose = self.robot_pose.copy()
            self.path_history.clear()
            self.planned_path = []
            self.last_scan_points = None
            self.frontiers.clear()
            self.exploration_targets.clear()
    
    def world_to_map(self, world_coords):
        """世界坐标转地图坐标"""
        map_coords = (world_coords[:2] / MAP_RESOLUTION + self.origin).astype(int)
        return np.clip(map_coords, 0, MAP_SIZE-1)
    
    def map_to_world(self, map_coords):
        """地图坐标转世界坐标"""
        return (map_coords - self.origin) * MAP_RESOLUTION
    
    def update_map(self, point_cloud):
        """使用概率栅格建图算法更新地图"""
        if not point_cloud:
            return
            
        try:
            # 转换为numpy数组并过滤
            points = np.array([[p['x'], p['y']] for p in point_cloud])
            mask = (np.abs(points[:, 0]) < 5.0) & (np.abs(points[:, 1]) < 5.0)
            points = points[mask]
            
            if len(points) == 0:
                return
                
            # ICP匹配进行位姿优化
            if self.last_scan_points is not None and len(self.last_scan_points) > 10:
                pose_correction = self.icp_match(points, self.last_scan_points)
                if pose_correction is not None:
                    self.robot_pose[:2] += pose_correction[:2]
                    self.robot_pose[2] += pose_correction[2]
            
            self.last_scan_points = points.copy()
            
            # 概率栅格建图
            with self.map_lock:
                self._update_probabilistic_map(points)
                self._update_frontiers()
                
        except Exception as e:
            print(f"地图更新错误: {e}")
    
    def _update_probabilistic_map(self, points):
        """使用概率栅格建图算法"""
        robot_pos = self.robot_pose[:2]
        robot_map_pos = self.world_to_map(robot_pos)
        
        # 更新空闲区域 (射线追踪)
        for point in points:
            end_pos = self.world_to_map(point)
            self._ray_tracing(robot_map_pos, end_pos, is_free=True)
        
        # 更新占用区域
        map_points = self.world_to_map(points)
        for x, y in map_points:
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                self.map_log_odds[y, x] += self.log_occ - np.log(self.l_prior/(1-self.l_prior))
        
        # 限制log-odds范围避免数值不稳定
        self.map_log_odds = np.clip(self.map_log_odds, -10, 10)
        
        # 转换为概率
        self.map_probabilities = 1.0 / (1.0 + np.exp(-self.map_log_odds))
        
        # 平滑处理
        self.map_probabilities = gaussian_filter(self.map_probabilities, sigma=0.5)
    
    def _ray_tracing(self, start, end, is_free=True):
        """Bresenham射线追踪算法"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if x < 0 or x >= MAP_SIZE or y < 0 or y >= MAP_SIZE:
                break
                
            if is_free:
                # 更新空闲区域的log-odds
                self.map_log_odds[y, x] += self.log_free - np.log(self.l_prior/(1-self.l_prior))
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def icp_match(self, current_points, last_points):
        """简化的ICP匹配算法"""
        try:
            if len(current_points) < 10 or len(last_points) < 10:
                return None
                
            # 使用最近邻匹配 (简化版)
            current_centroid = np.mean(current_points, axis=0)
            last_centroid = np.mean(last_points, axis=0)
            
            # 计算相对位移
            delta = current_centroid - last_centroid
            
            # 限制位移范围
            if np.linalg.norm(delta) < 0.2:  # 最大位移限制
                return np.array([delta[0], delta[1], 0.0])
            
            return None
            
        except Exception:
            return None
    
    def _update_frontiers(self):
        """更新前沿区域用于探索"""
        frontiers = []
        
        # 寻找未知与已知空闲区域的边界
        unknown_mask = (self.map_probabilities > 0.3) & (self.map_probabilities < 0.7)
        free_mask = self.map_probabilities < 0.3
        
        # 使用形态学操作找到边界
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_edges = cv2.Canny(unknown_mask.astype(np.uint8) * 255, 50, 150)
        
        # 找到潜在的探索目标
        contours, _ = cv2.findContours(unknown_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # 过滤小区域
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if free_mask[cy, cx]:  # 确保在已知空闲区域
                        frontiers.append([cx, cy])
        
        self.frontiers = frontiers
    
    def is_valid_position(self, map_pos):
        """检查位置是否有效(无碰撞)"""
        x, y = map_pos
        if x < 0 or x >= MAP_SIZE or y < 0 or y >= MAP_SIZE:
            return False
            
        x_min = max(0, x - self.safety_pixels)
        x_max = min(MAP_SIZE-1, x + self.safety_pixels)
        y_min = max(0, y - self.safety_pixels)
        y_max = min(MAP_SIZE-1, y + self.safety_pixels)
        
        return np.max(self.map[y_min:y_max+1, x_min:x_max+1]) < self.obstacle_threshold
    
    def heuristic(self, a, b):
        """A*启发式函数"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def a_star_path_planning(self, start_pos, goal_pos, dynamic_obstacles=True):
        """增强A*路径规划，支持动态避障"""
        start = tuple(self.world_to_map(start_pos))
        goal = tuple(self.world_to_map(goal_pos))
        
        # 动态权重调整
        dynamic_weight = 1.0
        if dynamic_obstacles:
            # 根据周围障碍物密度调整启发式权重
            obstacle_density = self._get_obstacle_density(start)
            dynamic_weight = 1.0 + obstacle_density * 0.5
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal) * dynamic_weight}
        
        # 8方向移动 + 更精细的移动代价
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), 
                    (1,1), (1,-1), (-1,1), (-1,-1)]
        
        closed_set = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            if current == goal:
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
                
                # 动态移动代价
                move_cost = 1.4 if dx and dy else 1.0
                
                # 根据障碍物概率调整代价
                obstacle_prob = self.map_probabilities[neighbor[1], neighbor[0]]
                if obstacle_prob > 0.5:
                    move_cost *= (1.0 + obstacle_prob * 2.0)
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, goal) * dynamic_weight
                    f_score[neighbor] = tentative_g + h_score
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _get_obstacle_density(self, pos, radius=5):
        """计算指定位置周围的障碍物密度"""
        x, y = pos
        x_min = max(0, x - radius)
        x_max = min(MAP_SIZE-1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(MAP_SIZE-1, y + radius)
        
        region = self.map_probabilities[y_min:y_max+1, x_min:x_max+1]
        obstacle_count = np.sum(region > self.obstacle_threshold)
        total_cells = region.size
        
        return obstacle_count / total_cells if total_cells > 0 else 0
    
    def get_obstacle_direction(self):
        """检测障碍物方向"""
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
                dx = self.target_position[0] - self.robot_pose[0]
                dy = self.target_position[1] - self.robot_pose[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.3:
                    sport_client.StopMove()
                    self.navigating = False
                    break
                
                path = self.a_star_path_planning(self.robot_pose[:2], self.target_position)
                if not path:
                    print("无法找到路径")
                    self.navigating = False
                    break
                
                self.planned_path = path
                
                lookahead_dist = 1.0
                next_point = None
                for point in path:
                    dist = math.sqrt((point[0]-self.robot_pose[0])**2 + 
                                   (point[1]-self.robot_pose[1])**2)
                    if dist >= lookahead_dist:
                        next_point = point
                        break
                
                if next_point is None:
                    next_point = path[-1]
                
                target_angle = math.atan2(next_point[1]-self.robot_pose[1], 
                                         next_point[0]-self.robot_pose[0])
                angle_diff = (target_angle - self.robot_pose[2] + math.pi) % (2*math.pi) - math.pi
                
                speed = min(0.5, distance)
                turn_rate = np.clip(angle_diff, -0.5, 0.5)
                
                obstacle_dirs = self.get_obstacle_direction()
                if 'front' in obstacle_dirs and distance < 1.0:
                    speed *= 0.5
                    if abs(angle_diff) > math.pi/4:
                        speed = 0.1
                
                sport_client.Move(speed, 0, turn_rate)
                time.sleep(0.1)
                
        finally:
            sport_client.StopMove()
            self.navigating = False
    
    def explore_environment(self, sport_client, duration=MAX_EXPLORE_TIME):
        """自动探索环境"""
        if self.exploring:
            return False
            
        self.exploring = True
        
        self.explore_thread = threading.Thread(
            target=self._explore_worker, 
            args=(sport_client, duration),
            daemon=True
        )
        self.explore_thread.start()
        
        self.save_thread = threading.Thread(
            target=self._auto_save_map_worker,
            daemon=True
        )
        self.save_thread.start()
        
        return True
    
    def _explore_worker(self, sport_client, duration):
        """前沿驱动探索工作线程"""
        start_time = time.time()
        current_target = None
        stuck_timer = 0
        last_position = self.robot_pose.copy()
        
        try:
            while self.exploring and (time.time() - start_time) < duration:
                # 检查是否卡住
                if np.linalg.norm(self.robot_pose[:2] - last_position[:2]) < 0.1:
                    stuck_timer += 1
                else:
                    stuck_timer = 0
                last_position = self.robot_pose.copy()
                
                # 如果卡住，随机转向
                if stuck_timer > 20:
                    sport_client.Move(0, 0, np.random.uniform(-1, 1))
                    time.sleep(1.0)
                    stuck_timer = 0
                    continue
                
                # 寻找最近的探索前沿
                if not current_target or self._should_select_new_target(current_target):
                    current_target = self._select_next_exploration_target()
                    if current_target is None:
                        # 没有更多前沿，随机探索
                        current_target = self._generate_random_exploration_target()
                
                if current_target is not None:
                    # 导航到探索目标
                    success = self._navigate_to_exploration_target(sport_client, current_target)
                    if not success:
                        # 如果无法到达，标记为已探索
                        self._mark_target_explored(current_target)
                        current_target = None
                
                # 更新机器人位姿 (基于实际运动)
                self._update_pose_from_movement()
                
                time.sleep(0.1)
                
        finally:
            sport_client.StopMove()
            self.exploring = False
    
    def _select_next_exploration_target(self):
        """选择下一个探索目标"""
        if not self.frontiers:
            return None
        
        robot_pos = self.robot_pose[:2]
        robot_map_pos = self.world_to_map(robot_pos)
        
        # 计算到每个前沿的信息增益和距离
        best_target = None
        best_score = -float('inf')
        
        for frontier in self.frontiers:
            frontier_world = self.map_to_world(np.array(frontier))
            
            # 计算距离
            distance = np.linalg.norm(frontier_world - robot_pos)
            if distance < 0.5 or distance > 3.0:  # 过滤太近或太远的目标
                continue
            
            # 检查可达性
            if not self._is_reachable(frontier):
                continue
            
            # 计算信息增益 (简化版)
            info_gain = self._calculate_info_gain(frontier)
            
            # 综合评分
            score = info_gain / (distance + 0.1)
            
            if score > best_score:
                best_score = score
                best_target = frontier_world
        
        return best_target
    
    def _is_reachable(self, map_pos):
        """检查目标是否可达"""
        return self.is_valid_position(map_pos)
    
    def _calculate_info_gain(self, frontier):
        """计算探索目标的信息增益"""
        x, y = frontier
        radius = 3
        x_min = max(0, x - radius)
        x_max = min(MAP_SIZE-1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(MAP_SIZE-1, y + radius)
        
        # 计算未知区域的数量
        region = self.map_probabilities[y_min:y_max+1, x_min:x_max+1]
        unknown_count = np.sum((region > 0.3) & (region < 0.7))
        
        return unknown_count
    
    def _generate_random_exploration_target(self):
        """生成随机探索目标"""
        robot_pos = self.robot_pose[:2]
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(1.0, 2.0)
        
        target_x = robot_pos[0] + distance * np.cos(angle)
        target_y = robot_pos[1] + distance * np.sin(angle)
        
        target = np.array([target_x, target_y])
        return target
    
    def _should_select_new_target(self, current_target):
        """判断是否需要选择新的探索目标"""
        robot_pos = self.robot_pose[:2]
        distance = np.linalg.norm(current_target - robot_pos)
        return distance < 0.3  # 如果已经很接近目标
    
    def _navigate_to_exploration_target(self, sport_client, target):
        """导航到探索目标"""
        robot_pos = self.robot_pose[:2]
        
        # 使用A*路径规划
        path = self.a_star_path_planning(robot_pos, target)
        if not path:
            return False
        
        # 简化导航
        if len(path) > 0:
            next_point = path[0]
            dx = next_point[0] - robot_pos[0]
            dy = next_point[1] - robot_pos[1]
            
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 0.1:
                target_angle = np.arctan2(dy, dx)
                angle_diff = (target_angle - self.robot_pose[2] + np.pi) % (2*np.pi) - np.pi
                
                # 动态速度调整
                speed = min(0.3, distance * 0.5)
                turn_rate = np.clip(angle_diff, -0.5, 0.5)
                
                # 避障调整
                obstacle_dirs = self.get_obstacle_direction()
                if 'front' in obstacle_dirs and speed > 0.1:
                    speed = 0.1
                
                sport_client.Move(speed, 0, turn_rate)
                return True
        
        return False
    
    def _mark_target_explored(self, target):
        """标记探索目标为已探索"""
        if target is not None:
            target_map = self.world_to_map(target)
            if 0 <= target_map[0] < MAP_SIZE and 0 <= target_map[1] < MAP_SIZE:
                # 降低该区域的信息增益
                x, y = target_map
                radius = 2
                x_min = max(0, x - radius)
                x_max = min(MAP_SIZE-1, x + radius)
                y_min = max(0, y - radius)
                y_max = min(MAP_SIZE-1, y + radius)
                
                self.map_log_odds[y_min:y_max+1, x_min:x_max+1] += 0.5
    
    def _update_pose_from_movement(self):
        """根据实际运动更新位姿"""
        # 简化的位姿更新，实际应用中应使用里程计数据
        pass
    
    def stop_exploration(self):
        """停止探索"""
        self.exploring = False
        if self.explore_thread:
            self.explore_thread.join(timeout=1.0)
        if hasattr(self, 'save_thread') and self.save_thread:
            self.save_thread.join(timeout=1.0)
    
    def save_map(self, filename=None):
        """保存增强地图数据"""
        if filename is None:
            filename = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
        filepath = os.path.join(MAPS_DIR, filename)
        
        with self.map_lock:
            map_data = {
                'map_log_odds': self.map_log_odds.copy(),
                'map_probabilities': self.map_probabilities.copy(),
                'resolution': MAP_RESOLUTION,
                'origin': self.origin,
                'robot_pose': self.robot_pose,
                'path_history': list(self.path_history),
                'frontiers': self.frontiers,
                'exploration_targets': self.exploration_targets,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(map_data, f)
                
            self._save_map_image()
                
        print(f"增强地图已保存到 {filepath}")
        return filepath
    
    def _save_map_image(self):
        """保存地图图片"""
        map_img = self.get_map_image()
        if map_img is not None:
            img_path = os.path.join("static/photos", "slam_map.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(map_img, cv2.COLOR_RGBA2BGR))
    
    def load_map(self, filename):
        """加载增强地图数据"""
        filepath = os.path.join(MAPS_DIR, filename)
        
        try:
            with open(filepath, 'rb') as f:
                map_data = pickle.load(f)
                
            with self.map_lock:
                self.map_log_odds = map_data.get('map_log_odds', np.zeros((MAP_SIZE, MAP_SIZE)))
                self.map_probabilities = map_data.get('map_probabilities', 
                                                    np.full((MAP_SIZE, MAP_SIZE), 0.5))
                self.origin = map_data['origin']
                self.robot_pose = map_data['robot_pose']
                self.path_history = deque(map_data.get('path_history', []), maxlen=1000)
                self.frontiers = map_data.get('frontiers', [])
                self.exploration_targets = map_data.get('exploration_targets', [])
                
            self._save_map_image()
                
            print(f"已加载增强地图 {filename}")
            return True
            
        except Exception as e:
            print(f"加载地图失败: {e}")
            return False
    
    def get_map_image(self):
        """获取增强的地图可视化图像"""
        with self.map_lock:
            # 创建彩色地图
            colored_map = np.zeros((MAP_SIZE, MAP_SIZE, 4), dtype=np.float32)
            
            # 使用概率值创建颜色映射
            prob = self.map_probabilities
            
            # 障碍物: 黑色, 空闲: 白色, 未知: 灰色
            colored_map[:, :, 0] = 1.0 - prob  # Red channel
            colored_map[:, :, 1] = 1.0 - prob  # Green channel
            colored_map[:, :, 2] = 1.0 - prob  # Blue channel
            colored_map[:, :, 3] = 1.0  # Alpha channel
            
            # 绘制路径历史
            if len(self.path_history) > 1:
                path = np.array(self.path_history)
                path_map = self.world_to_map(path[:, :2])
                valid_points = [(x, y) for x, y in path_map 
                              if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE]
                for x, y in valid_points:
                    colored_map[y, x] = [1.0, 0.0, 0.0, 1.0]  # 红色路径
            
            # 绘制规划路径
            if self.planned_path:
                path = np.array(self.planned_path)
                path_map = self.world_to_map(path[:, :2])
                valid_points = [(x, y) for x, y in path_map 
                              if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE]
                for x, y in valid_points:
                    colored_map[y, x] = [0.0, 1.0, 0.0, 1.0]  # 绿色路径
            
            # 绘制前沿区域
            for frontier in self.frontiers:
                fx, fy = frontier
                if 0 <= fx < MAP_SIZE and 0 <= fy < MAP_SIZE:
                    colored_map[fy, fx] = [1.0, 1.0, 0.0, 1.0]  # 黄色前沿
                    
            # 绘制机器人位置
            robot_pos = self.world_to_map(self.robot_pose[:2])
            if 0 <= robot_pos[0] < MAP_SIZE and 0 <= robot_pos[1] < MAP_SIZE:
                cv2.circle(colored_map, (robot_pos[0], robot_pos[1]), 4, (0, 0, 1.0), -1)
                
                # 绘制机器人朝向
                end_x = int(robot_pos[0] + 8 * np.cos(self.robot_pose[2]))
                end_y = int(robot_pos[1] + 8 * np.sin(self.robot_pose[2]))
                if 0 <= end_x < MAP_SIZE and 0 <= end_y < MAP_SIZE:
                    cv2.line(colored_map, (robot_pos[0], robot_pos[1]), 
                            (end_x, end_y), (0, 0, 1.0), 2)
            
            # 绘制目标位置
            if self.target_position is not None:
                target_pos = self.world_to_map(self.target_position[:2])
                if 0 <= target_pos[0] < MAP_SIZE and 0 <= target_pos[1] < MAP_SIZE:
                    cv2.circle(colored_map, (target_pos[0], target_pos[1]), 6, (1.0, 0.0, 1.0), 2)
            
            # 旋转地图以匹配机器人朝向
            angle = np.degrees(self.robot_pose[2])
            rotated_map = rotate(colored_map, angle, center=(robot_pos[0], robot_pos[1]), order=0)
            
            return (rotated_map * 255).astype(np.uint8)
    
    def update_pose(self, dx, dy, dtheta):
        """更新机器人位姿"""
        with self.map_lock:
            self.robot_pose[0] += dx
            self.robot_pose[1] += dy
            self.robot_pose[2] = (self.robot_pose[2] + dtheta) % (2 * np.pi)
            self.path_history.append(self.robot_pose.copy())
    
    def _auto_save_map_worker(self):
        """自动保存地图线程"""
        while self.exploring or self.navigating:
            try:
                self._save_map_image()
                time.sleep(1.0)
            except Exception as e:
                print(f"自动保存地图错误: {e}")
                time.sleep(1.0)

# ============================================================
#  Lidar 处理器
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
        self.slam_mapper = slam_mapper
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
            
            if self.slam_mapper and point_cloud:
                self.slam_mapper.update_map(point_cloud)
                
            self._process_point_cloud(cloud_data)

    def _process_point_cloud(self, cloud_data: PointCloud2_):
        try:
            if not cloud_data.fields or len(cloud_data.data) == 0:
                raise ValueError("Empty point cloud data")
    
            grid_cols, grid_rows = 60, 20
            grid = [["·" for _ in range(grid_cols)] for _ in range(grid_rows)]
    
            cloud_bytes = bytes(cloud_data.data)
    
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
    
            for axis in ["x", "y", "z"]:
                if axis not in field_config:
                    raise ValueError(f"Missing coordinate field: {axis}")
    
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
    
            self._save_point_cloud_image(valid_points)
    
            self.obstacle_distance = (
                min(p['x'] for p in valid_points) if valid_points else None
            self.scan_complete = True
    
        except Exception as exc:
            print(f"点云处理异常: {type(exc).__name__}: {exc}")
            self.obstacle_distance = None
            self.scan_complete = False

    def _save_point_cloud_image(self, points):
        """保存点云图像"""
        if not points:
            return
        try:
            arr = np.array([[p["x"], p["y"], p["z"]] for p in points])
            
            distances = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
            
            color_map = [
                (0.0, 0.5, [1.0, 0.0, 0.0]),
                (0.5, 1.0, [1.0, 0.5, 0.0]),
                (1.0, 1.5, [1.0, 1.0, 0.0]),
                (1.5, 2.0, [0.7, 1.0, 0.0]),
                (2.0, 2.5, [0.0, 1.0, 0.0]),
                (2.5, 3.0, [0.0, 1.0, 0.7]),
                (3.0, float('inf'), [0.0, 0.0, 1.0])
            ]
            
            colors = []
            for d in distances:
                for min_d, max_d, color in color_map:
                    if min_d <= d < max_d:
                        colors.append(color)
                        break
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            
            ax.scatter(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                c=colors,
                s=2,
                alpha=0.7
            )
            
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("Point Cloud (Color by 0.5m Distance Intervals)")
            
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
            
            ax.view_init(elev=25, azim=45)
            plt.tight_layout()
            
            plt.savefig("static/photos/point_cloud.jpg", dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as exc:
            print(f"点云图片保存失败: {exc}")

    def _lidar_worker(self):
        while self.lidar_running:
            time.sleep(0.1)

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
        
        self.app.add_url_rule("/maps", "get_maps", self.get_maps, methods=["GET"])
        self.app.add_url_rule("/map/<filename>", "get_map", self.get_map)
        self.app.add_url_rule("/slam_map", "get_slam_map", self.get_slam_map)

        self.auto_photo_enabled = True
        self.photo_interval = 3000  # ms

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

        self._start_auto_photo()

        self.server_thread = threading.Thread(
            target=self.run_server, daemon=True
        )
        self.server_thread.start()

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

    def send_tts_message(self, message):
        self.socketio.emit("tts_message", {"message": message})

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
        return send_from_directory(MAPS_DIR, filename)
    
    def get_slam_map(self):
        if hasattr(self.voice_controller, "slam_mapper"):
            map_image = self.voice_controller.slam_mapper.get_map_image()
            if map_image is not None:
                map_path = "static/photos/slam_map.jpg"
                cv2.imwrite(map_path, cv2.cvtColor(map_image, cv2.COLOR_RGBA2BGR))
                
                return jsonify({
                    "url": "/photo/slam_map.jpg",
                    "timestamp": time.time()
                })
        return jsonify({"error": "No map available"}), 404
        
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
#  YOLOv5 检测器
# ============================================================
class OfflineYOLODetector:
    """离线 YOLOv5 目标检测器"""

    def __init__(self, repo_path="yolov5", model_weights="yolov5s.pt"):
        self.model = None
        self.class_names = None
        print("开始加载 YOLOv5 模型...")
        self._init_detector(repo_path, model_weights)
        print("YOLOv5 模型加载完成")

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
#  主控制类
# ============================================================
class VoiceControl:
    def __init__(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.cmd_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.asr_queue = queue.Queue()

        self.loop = asyncio.new_event_loop()
        self.tts_thread = None

        self.running = True
        self.debug_mode = True

        self.detector = OfflineYOLODetector(
            repo_path="yolov5", model_weights="yolov5s.pt"
        )
        self.last_photo_path = ""

        self.current_network_status = self.check_network()

        self.web_interface = WebInterface(self)

        self.audio_interface = pyaudio.PyAudio()
        self.init_voice_engine()

        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

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
        }

        self.slam_mapper = SLAMMapper()
        self.lidar_processor = LidarProcessor(self.slam_mapper)
        self.lidar_processor.start_lidar()
        
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.obstacle_client = ObstaclesAvoidClient()
        self.obstacle_client.SetTimeout(10.0)
        self.obstacle_client.Init()
        self.obstacle_client.SwitchSet(True)
        self.obstacle_client.UseRemoteCommandFromApi(True)

        print("初始化完成\n")

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

    def signal_handler(self, signum, frame):
        print("\n收到终止信号，正在退出...")
        self.running = False
        self.cleanup_resources()

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

    def check_network(self):
        return False
        try:
            socket.create_connection(("www.baidu.com", 80), timeout=2)
            return True
        except OSError:
            return False

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

    def network_monitor(self):
        while self.running:
            new_status = self.check_network()
            if new_status != self.current_network_status:
                self.current_network_status = new_status
                self.tts_queue.put(
                    f"网络状态已切换至{'在线' if new_status else '离线'}模式"
                )
            time.sleep(5)

    def process_command(self, command):
        cmd = command.lower().replace(" ", "")
        print(f"{cmd}")

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

        action_id = None
        for pattern, cmd_id in self.command_map.items():
            if re.search(pattern, cmd):
                action_id = cmd_id
                break

        if action_id is not None:
            self.execute_action(action_id, cmd)
        else:
            self.tts_queue.put("指令未识别")

    def execute_action(self, action_id, raw_cmd=""):
        print(f"执行动作 {action_id} ({raw_cmd})")
        try:
            if action_id == 10:
                if self.lidar_processor.scan_environment():
                    info = self.lidar_processor.get_point_cloud_info()
                    self.tts_queue.put(
                        f"环境扫描完成，获取 {info['data_size']} 个数据点"
                    )
                else:
                    self.tts_queue.put("环境扫描失败")
                return

            if action_id == 11:
                if self.lidar_processor.scan_environment():
                    dist = self.lidar_processor.get_obstacle_distance()
                    if dist is not None:
                        self.tts_queue.put(f"前方 {dist:.1f} 米处检测到障碍物")
                    else:
                        self.tts_queue.put("前方未检测到障碍物")
                else:
                    self.tts_queue.put("障碍物检测失败")
                return

            if action_id == 12:
                dist = self.lidar_processor.get_obstacle_distance()
                if dist is not None:
                    self.tts_queue.put(f"前方 {dist:.1f} 米处有障碍物")
                else:
                    self.tts_queue.put("前方暂无障碍物")
                return

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
                
            elif action_id == 20:
                if self.slam_mapper.explore_environment(self.sport_client):
                    self.tts_queue.put("开始建图，机器人将自动探索环境")
                else:
                    self.tts_queue.put("建图已在运行中")
                    
            elif action_id == 21:
                self.slam_mapper.stop_exploration()
                self.tts_queue.put("已停止建图")
                
            elif action_id == 22:
                map_path = self.slam_mapper.save_map()
                self.tts_queue.put(f"地图已保存到 {os.path.basename(map_path)}")
                
            elif action_id == 23:
                maps = [f for f in os.listdir(MAPS_DIR) if f.endswith('.pkl')]
                if maps:
                    latest_map = max(maps, key=lambda f: os.path.getmtime(os.path.join(MAPS_DIR, f)))
                    if self.slam_mapper.load_map(latest_map):
                        self.tts_queue.put(f"已加载地图 {latest_map}")
                    else:
                        self.tts_queue.put("加载地图失败")
                else:
                    self.tts_queue.put("没有找到地图文件")
                    
            elif action_id == 24:
                self.tts_queue.put("正在显示当前地图")
        
            if not self.debug_mode and action_id not in (7, 8, 10, 11, 12):
                self.tts_queue.put("指令已执行")

        except Exception as exc:
            err = f"执行出错: {exc}"
            self.tts_queue.put(err)
            print(err)

    def handle_take_photo(self):
        code, data = self.video_client.GetImageSample()
        if code == 0:
            self.last_photo_path = self.web_interface.save_photo(bytes(data))
            self.tts_queue.put("拍照成功")
            self.handle_image_analysis()
        else:
            self.tts_queue.put("拍照失败，请重试")

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

    def cleanup_resources(self):
        print("正在清理资源...")
        self.running = False
        
        if hasattr(self, "slam_mapper"):
            self.slam_mapper.stop_exploration()
            
        if hasattr(self, "audio_stream") and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        if hasattr(self, "audio_stream"):
            self.audio_stream.close()
        if hasattr(self, "audio_interface"):
            self.audio_interface.terminate()

        try:
            self.sport_client.StandDown()
            time.sleep(2)
            self.sport_client.StopMove()
        except Exception:
            pass

        try:
            del self.video_client
        except Exception:
            pass

        try:
            self.lidar_processor.stop_lidar()
        except Exception:
            pass

        if self.detector and self.detector.model:
            del self.detector.model
            torch.cuda.empty_cache()

        if self.loop.is_running():
            self.loop.stop()

        for q in (self.tts_queue, self.asr_queue):
            while not q.empty():
                q.get_nowait()

    def run(self):
        try:
            self.execute_action(1)
            self.tts_queue.put("系统已启动，准备就绪")

            self.tts_thread = threading.Thread(
                target=self.loop.run_forever, daemon=True
            )
            self.tts_thread.start()
            asyncio.run_coroutine_threadsafe(self.tts_worker(), self.loop)

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
