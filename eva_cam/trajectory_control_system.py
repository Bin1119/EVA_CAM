#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动轨迹控制系统
提供预设运动轨迹选择和同步数据采集功能
"""

import sys
import os
import time
import threading
import cv2 as cv
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_cam_controller import EvaCamController
from utils import logger, setup_paths

# Setup paths for AlpLib
setup_paths()

# Add AlpLib bin directory to path for AlpPython module
alplib_bin_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alplib', 'bin')
sys.path.append(alplib_bin_path)

try:
    from AlpPython import *
except ImportError as e:
    logger.error(f"Failed to import AlpPython: {e}")
    sys.exit(1)


class TrajectoryControlSystem:
    """运动轨迹控制系统"""
    
    def __init__(self):
        self.controller = EvaCamController()
        self.is_running = False
        self.current_position = None
        self.initial_position = None
        self.camera_display_thread = None
        self.aps_display_thread = None
        self.evs_display_thread = None
        self.display_running = False
        self.is_collecting_data = False
        self.data_collection_active = False
        self.default_distance = self._get_default_distance()
        self.default_rotation_angle = self._get_default_rotation_angle()
        self.linear_speed = self._get_linear_speed()
        self.vertical_speed = self._get_vertical_speed()
        self.rotation_speed = self._get_rotation_speed()
        self.trajectories = self._define_trajectories()
        
    def _get_default_distance(self) -> int:
        """从配置文件获取默认移动距离"""
        try:
            return int(self.controller.config.get('MOTION', 'default_distance_mm', fallback=150))
        except:
            return 150
        
    def _get_default_rotation_angle(self) -> int:
        """从配置文件获取默认旋转角度"""
        try:
            return int(self.controller.config.get('MOTION', 'default_rotation_angle_degrees', fallback=45))
        except:
            return 45
        
    def _get_linear_speed(self) -> int:
        """从配置文件获取线性移动速度"""
        try:
            return int(self.controller.config.get('MOTION', 'linear_speed', fallback=100))
        except:
            return 100
        
    def _get_vertical_speed(self) -> int:
        """从配置文件获取垂直移动速度"""
        try:
            return int(self.controller.config.get('MOTION', 'vertical_speed', fallback=100))
        except:
            return 100
        
    def _get_rotation_speed(self) -> int:
        """从配置文件获取旋转速度"""
        try:
            return int(self.controller.config.get('MOTION', 'rotation_speed', fallback=100))
        except:
            return 100
    
    def _start_motion_data_collection(self) -> bool:
        """在运动开始前启动数据采集"""
        try:
            if self.is_collecting_data:
                logger.warning("数据采集已经在进行中")
                return True
            
            if not hasattr(self.controller, 'camera') or not self.controller.camera:
                logger.error("相机未初始化，无法开始数据采集")
                return False
            
            # 获取保存格式配置
            recording_config = self.controller.config.get_recording_config()
            save_format = recording_config.get('format', 'hdf5').lower()
            
            # 使用已创建的输出目录
            if not hasattr(self.controller, 'current_output_dir') or not self.controller.current_output_dir:
                logger.error("输出目录未准备好")
                return False
            
            output_dir = self.controller.current_output_dir
            
            # 根据格式设置文件名和写入器类型
            if save_format == 'bin':
                file_name = os.path.join(output_dir, f"motion_data_{datetime.now().strftime('%Y%m%d%H%M%S')}_{datetime.now().microsecond // 1000:03d}")
                writer_type = WriterType.BIN
            else:  # hdf5
                file_name = os.path.join(output_dir, f"motion_data_{datetime.now().strftime('%Y%m%d%H%M%S')}_{datetime.now().microsecond // 1000:03d}.alpdata")
                writer_type = WriterType.HDF5
            
            # 获取设备属性
            device = self.controller.camera
            device_attribute = device.getDeviceAttribute(writer_type, file_name)
            
            # 获取APS和EVS属性
            aps_ptr = None
            evs_ptr = None
            
            if device.apsModeIndex() > -1:
                aps_ptr = device.getApsAttribute()
            
            if device.evsModeIndex() > -1:
                evs_ptr = device.getEvsAttribute()
            
            # 打开存储文件
            error = self.controller.writer_file.open(device_attribute, aps_ptr, evs_ptr)
            
            if error != AlpSaveFileError.NONE:
                logger.error(f"打开存储文件失败: {error}")
                return False
            
            self.is_collecting_data = True
            
            logger.info(f"运动数据采集已开始，保存格式: {save_format.upper()}")
            logger.info(f"输出文件: {file_name}")
            
            # 更新轨迹信息文件，记录数据采集开始
            if hasattr(self.controller, 'current_output_dir') and self.controller.current_output_dir:
                info_file = os.path.join(self.controller.current_output_dir, "trajectory_info.txt")
                if os.path.exists(info_file):
                    with open(info_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n数据采集开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                        f.write(f"数据文件: {os.path.basename(file_name)}\n")
                        f.write("状态: 运动中...\n")
            
            return True
            
        except Exception as e:
            logger.error(f"启动运动数据采集失败: {e}")
            return False
    
    def _stop_motion_data_collection(self):
        """在运动完成后停止数据采集"""
        try:
            if not self.is_collecting_data:
                return
            
            if hasattr(self.controller, 'writer_file') and self.controller.writer_file:
                self.controller.writer_file.close()
            
            self.is_collecting_data = False
            logger.info("运动数据采集已停止")
            
            # 更新轨迹信息文件，记录数据采集结束
            if hasattr(self.controller, 'current_output_dir') and self.controller.current_output_dir:
                info_file = os.path.join(self.controller.current_output_dir, "trajectory_info.txt")
                if os.path.exists(info_file):
                    with open(info_file, 'a', encoding='utf-8') as f:
                        f.write(f"数据采集结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                        f.write("状态: 运动完成\n")
            
        except Exception as e:
            logger.error(f"停止运动数据采集失败: {e}")
        
    def _define_trajectories(self) -> Dict[str, Dict[str, Any]]:
        """定义预设运动轨迹"""
        return {
            '1': {
                'name': f'X+方向移动{self.default_distance}毫米',
                'description': f'机械臂沿X+方向移动{self.default_distance}毫米',
                'motion_func': self._move_horizontal_right,
                'params': {'distance_mm': self.default_distance}
            },
            '2': {
                'name': f'X-方向移动{self.default_distance}毫米',
                'description': f'机械臂沿X-方向移动{self.default_distance}毫米',
                'motion_func': self._move_horizontal_left,
                'params': {'distance_mm': self.default_distance}
            },
            '3': {
                'name': f'Z+方向移动{self.default_distance}毫米',
                'description': f'机械臂沿Z+方向移动{self.default_distance}毫米',
                'motion_func': self._move_vertical_up,
                'params': {'distance_mm': self.default_distance}
            },
            '4': {
                'name': f'Z-方向移动{self.default_distance}毫米',
                'description': f'机械臂沿Z-方向移动{self.default_distance}毫米',
                'motion_func': self._move_vertical_down,
                'params': {'distance_mm': self.default_distance}
            },
            '5': {
                'name': f'Y+方向移动{self.default_distance}毫米',
                'description': f'机械臂沿Y+方向移动{self.default_distance}毫米',
                'motion_func': self._move_y_positive,
                'params': {'distance_mm': self.default_distance}
            },
            '6': {
                'name': f'Y-方向移动{self.default_distance}毫米',
                'description': f'机械臂沿Y-方向移动{self.default_distance}毫米',
                'motion_func': self._move_y_negative,
                'params': {'distance_mm': self.default_distance}
            },
            '7': {
                'name': f'逆时针旋转{self.default_rotation_angle}度',
                'description': f'机械臂水平逆时针旋转{self.default_rotation_angle}度',
                'motion_func': self._rotate_counter_clockwise,
                'params': {'angle_degrees': self.default_rotation_angle}
            },
            '8': {
                'name': f'顺时针旋转{self.default_rotation_angle}度',
                'description': f'机械臂水平顺时针旋转{self.default_rotation_angle}度',
                'motion_func': self._rotate_clockwise,
                'params': {'angle_degrees': self.default_rotation_angle}
            },
            '9': {
                'name': '回到初始位置',
                'description': '机械臂回到系统启动时的初始位置',
                'motion_func': self._move_to_initial_position,
                'params': {}
            }
        }
    
    def confirm_startup(self) -> bool:
        """确认是否启动系统"""
        print("\n" + "="*60)
        print("          EVA_CAM 运动轨迹控制系统")
        print("="*60)
        print("\n系统功能：")
        print("• APS和EVS模式实时相机显示（双窗口）")
        print("• 预设运动轨迹选择")
        print("• 运动与数据采集同步")
        print("• 自动保存采集数据（支持HDF5和BIN格式）")
        print("• 实时位置信息显示")
        print("• 数据采集状态指示")
        print("• 默认移动距离可配置（在settings.conf的MOTION部分）")
        print("• 默认旋转角度可配置（在settings.conf的MOTION部分）")
        print("\n警告：请确保机械臂工作区域内无障碍物！")
        print("\n初始化流程：")
        print("1. 初始化相机系统")
        print("2. 验证APS和EVS数据采集")
        print("3. 确认数据正常后初始化机械臂")
        print("="*60)
        
        while True:
            try:
                choice = input("\n是否启动系统？ (y/n): ").strip().lower()
                if choice in ['y', 'yes', '是']:
                    logger.info("用户确认启动系统")
                    return True
                elif choice in ['n', 'no', '否']:
                    logger.info("用户取消启动系统")
                    return False
                else:
                    print("请输入 y/yes/是 或 n/no/否")
            except KeyboardInterrupt:
                print("\n检测到中断信号，退出系统")
                return False
    
    def initialize_systems(self) -> bool:
        """初始化系统"""
        try:
            logger.info("=== 初始化运动轨迹控制系统 ===")
            
            # 首先初始化相机
            if not self.controller.initialize_camera():
                logger.error("相机初始化失败")
                return False
            
            # 验证相机数据采集是否正常
            if not self._validate_camera_data_acquisition():
                logger.error("相机数据采集验证失败")
                return False
            
            # 相机数据采集正常后，再初始化机械臂
            if not self.controller.initialize_xarm(go_home=False):
                logger.error("机械臂初始化失败")
                return False
            
            # 获取当前位置
            position_data = self.controller.arm.get_position()
            if isinstance(position_data, tuple) and len(position_data) == 2:
                actual_position = position_data[1]
            elif isinstance(position_data, (list, tuple)) and len(position_data) >= 6:
                actual_position = position_data
            else:
                logger.error(f"位置数据格式不正确: {position_data}")
                return False
            
            if isinstance(actual_position, (list, tuple)) and len(actual_position) >= 6:
                self.current_position = list(actual_position)[:6]
                self.initial_position = list(actual_position)[:6]
            else:
                logger.error(f"实际位置数据格式不正确: {actual_position}")
                return False
            
            logger.info(f"初始位置: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, z={self.current_position[2]:.2f}")
            logger.info(f"默认移动距离: {self.default_distance}毫米")
            logger.info(f"默认旋转角度: {self.default_rotation_angle}度")
            logger.info("初始位置已保存，可用于返回初始位置功能")
            
            # 显示相机信息
            if hasattr(self.controller, 'camera') and self.controller.camera:
                device = self.controller.camera
                if device.apsModeIndex() > -1:
                    print(f"APS Info: {device.apsWidth()}x{device.apsHeight()}, {device.apsFps()}fps")
                if device.evsModeIndex() > -1:
                    print(f"EVS Info: {device.evsWidth()}x{device.evsHeight()}, {device.evsFps()}fps")
            
            logger.info("系统初始化完成！")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def _validate_camera_data_acquisition(self) -> bool:
        """验证相机数据采集是否正常工作"""
        try:
            logger.info("开始验证相机数据采集...")
            
            if not hasattr(self.controller, 'camera') or not self.controller.camera:
                logger.error("相机对象不存在")
                return False
            
            device = self.controller.camera
            
            # 检查相机是否打开
            if not device.isOpened():
                logger.error("相机未打开")
                return False
            
            # 验证APS数据采集
            aps_valid = False
            if device.apsModeIndex() > -1:
                logger.info("验证APS数据采集...")
                aps_valid = self._validate_aps_data(device)
                if not aps_valid:
                    logger.error("APS数据采集验证失败")
                    return False
                logger.info("APS数据采集验证成功")
            else:
                logger.warning("APS模式未启用，跳过APS验证")
            
            # 验证EVS数据采集
            evs_valid = False
            if device.evsModeIndex() > -1:
                logger.info("验证EVS数据采集...")
                evs_valid = self._validate_evs_data(device)
                if not evs_valid:
                    logger.error("EVS数据采集验证失败")
                    return False
                logger.info("EVS数据采集验证成功")
            else:
                logger.warning("EVS模式未启用，跳过EVS验证")
            
            # 至少需要一种数据采集模式正常工作
            if not aps_valid and not evs_valid:
                logger.error("APS和EVS数据采集均未正常工作")
                return False
            
            logger.info("相机数据采集验证完成")
            return True
            
        except Exception as e:
            logger.error(f"相机数据采集验证失败: {e}")
            return False
    
    def _validate_aps_data(self, device) -> bool:
        """验证APS数据采集"""
        try:
            # 尝试多次获取APS数据
            for attempt in range(5):  # 最多尝试5次
                frameslists = device.getApsFrames()
                
                if frameslists and len(frameslists) > 0:
                    # 验证帧数据
                    for frame in frameslists:
                        if hasattr(frame, 'convertTo'):
                            # 尝试转换帧数据
                            test_image = frame.convertTo()
                            if test_image is not None:
                                logger.info(f"APS数据获取成功 (尝试 {attempt + 1}/5)")
                                return True
                
                # 短暂等待后重试
                time.sleep(0.1)
            
            logger.error("无法获取有效的APS数据")
            return False
            
        except Exception as e:
            logger.error(f"APS数据验证失败: {e}")
            return False
    
    def _validate_evs_data(self, device) -> bool:
        """验证EVS数据采集"""
        try:
            # 尝试多次获取EVS数据
            for attempt in range(5):  # 最多尝试5次
                frameslists = device.getEvsFrames()
                
                if frameslists and len(frameslists) > 0:
                    # 验证帧数据
                    for frame in frameslists:
                        if hasattr(frame, 'frame'):
                            # 尝试获取帧数据
                            test_frame = frame.frame()
                            if test_frame is not None:
                                logger.info(f"EVS数据获取成功 (尝试 {attempt + 1}/5)")
                                return True
                
                # 短暂等待后重试
                time.sleep(0.1)
            
            logger.error("无法获取有效的EVS数据")
            return False
            
        except Exception as e:
            logger.error(f"EVS数据验证失败: {e}")
            return False
    
    def start_camera_display(self):
        """启动相机显示"""
        try:
            self.display_running = True
            self.camera_display_thread = threading.Thread(target=self._camera_display_loop)
            self.camera_display_thread.daemon = True
            self.camera_display_thread.start()
            logger.info("相机显示已启动")
        except Exception as e:
            logger.error(f"启动相机显示失败: {e}")
    
    def stop_camera_display(self):
        """停止相机显示"""
        try:
            logger.info("开始停止相机显示...")
            
            # 设置停止标志
            self.display_running = False
            
            # 停止数据采集
            if self.is_collecting_data:
                self.stop_data_collection()
            
            # 等待所有显示线程结束，增加超时时间和重试机制
            max_wait_time = 5  # 最大等待时间
            start_time = time.time()
            
            # 等待APS显示线程
            if self.aps_display_thread and self.aps_display_thread.is_alive():
                logger.info("等待APS显示线程结束...")
                self.aps_display_thread.join(timeout=max_wait_time)
                if self.aps_display_thread.is_alive():
                    logger.warning("APS显示线程未在超时时间内结束")
            
            # 等待EVS显示线程
            if self.evs_display_thread and self.evs_display_thread.is_alive():
                logger.info("等待EVS显示线程结束...")
                self.evs_display_thread.join(timeout=max_wait_time)
                if self.evs_display_thread.is_alive():
                    logger.warning("EVS显示线程未在超时时间内结束")
            
            # 等待主相机显示线程
            if self.camera_display_thread and self.camera_display_thread.is_alive():
                logger.info("等待主相机显示线程结束...")
                self.camera_display_thread.join(timeout=max_wait_time)
                if self.camera_display_thread.is_alive():
                    logger.warning("主相机显示线程未在超时时间内结束")
            
            # 清理线程引用
            self.aps_display_thread = None
            self.evs_display_thread = None
            self.camera_display_thread = None
            
            # 确保所有OpenCV窗口被关闭
            try:
                cv.destroyAllWindows()
                logger.info("OpenCV窗口已关闭")
            except Exception as e:
                logger.warning(f"关闭OpenCV窗口时出错: {e}")
            
            logger.info("相机显示已停止")
            
        except Exception as e:
            logger.error(f"停止相机显示失败: {e}")
    
    def _camera_display_loop(self):
        """相机显示循环 - 基于AlpLib示例代码的正确实现"""
        try:
            # 创建独立的APS和EVS显示线程
            self.aps_display_thread = threading.Thread(target=self._aps_display_loop)
            self.evs_display_thread = threading.Thread(target=self._evs_display_loop)
            
            self.aps_display_thread.daemon = True
            self.evs_display_thread.daemon = True
            
            self.aps_display_thread.start()
            self.evs_display_thread.start()
            
            logger.info("相机显示线程已启动")
            
            # 等待显示线程结束
            while self.display_running:
                time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"相机显示循环错误: {e}")
    
    def _aps_display_loop(self):
        """APS显示线程 - 基于AlpLib示例代码"""
        try:
            if not hasattr(self.controller, 'camera') or not self.controller.camera:
                return
            
            device = self.controller.camera
            
            # 只在APS模式启用时创建窗口
            if device.apsModeIndex() > -1:
                cv.namedWindow("APS Camera", cv.WINDOW_NORMAL)
                
                # 显示APS参数信息
                logger.info(f"APS Mode: {device.apsModeString()}")
                logger.info(f"APS Resolution: {device.apsWidth()}x{device.apsHeight()}")
                logger.info(f"APS FPS: {device.apsFps()}")
                logger.info(f"APS Exposure: {device.apsExposureTime()}us")
                logger.info(f"APS Gain: {device.apsAnalogGain()}")
            
            while self.display_running and device.isOpened():
                # 获取APS数据（基于AlpLib示例代码）
                frameslists = device.getApsFrames()
                
                # 保存数据（如果正在采集）
                if self.is_collecting_data and hasattr(self.controller, 'writer_file'):
                    for frame in frameslists:
                        self.controller.writer_file.write(frame)
                
                # 显示帧（基于AlpLib示例代码）
                if frameslists and device.apsModeIndex() > -1:
                    for frame in frameslists:
                        # 转换为OpenCV Mat格式
                        aps_image = frame.convertTo()
                        
                        # 添加时间戳和位置信息
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        status_text = "RECORDING" if self.is_collecting_data else "MONITOR"
                        cv.putText(aps_image, f"APS - {status_text} - {timestamp}", 
                                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.is_collecting_data else (255, 255, 255), 2)
                        
                        # 添加位置信息
                        if self.current_position:
                            pos_text = f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})"
                            cv.putText(aps_image, pos_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # 显示FPS信息
                        fps_text = f"FPS: {device.apsFps()}"
                        cv.putText(aps_image, fps_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv.imshow("APS Camera", aps_image)
                        cv.waitKey(1)
                        break  # 只处理第一帧
                
                # 控制处理频率
                time.sleep(0.001)  # 1ms间隔
            
            # 清理窗口
            if device.apsModeIndex() > -1:
                cv.destroyWindow("APS Camera")
            
        except Exception as e:
            logger.error(f"APS显示线程错误: {e}")
    
    def _evs_display_loop(self):
        """EVS显示线程 - 基于AlpLib示例代码"""
        try:
            if not hasattr(self.controller, 'camera') or not self.controller.camera:
                return
            
            device = self.controller.camera
            
            # 只在EVS模式启用时创建窗口
            if device.evsModeIndex() > -1:
                cv.namedWindow("EVS Camera", cv.WINDOW_NORMAL)
                
                # 显示EVS参数信息
                logger.info(f"EVS Mode: {device.evsModeString()}")
                logger.info(f"EVS Resolution: {device.evsWidth()}x{device.evsHeight()}")
                logger.info(f"EVS FPS: {device.evsFps()}")
                logger.info(f"EVS Sensitivity: {device.evsSensitivity()}")
            
            while self.display_running and device.isOpened():
                # 获取EVS数据（基于AlpLib示例代码）
                frameslists = device.getEvsFrames()
                
                # 保存数据（如果正在采集）
                if self.is_collecting_data and hasattr(self.controller, 'writer_file'):
                    for frame in frameslists:
                        self.controller.writer_file.write(frame)
                
                # 显示帧（基于AlpLib示例代码）
                if frameslists and device.evsModeIndex() > -1:
                    for frame in frameslists:
                        # 转换为OpenCV格式（基于AlpLib示例：evs_image * 100）
                        evs_image = frame.frame() * 100
                        
                        # 添加时间戳和位置信息
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        status_text = "RECORDING" if self.is_collecting_data else "MONITOR"
                        cv.putText(evs_image, f"EVS - {status_text} - {timestamp}", 
                                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.is_collecting_data else (255, 255, 255), 2)
                        
                        # 添加位置信息
                        if self.current_position:
                            pos_text = f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})"
                            cv.putText(evs_image, pos_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # 显示FPS信息
                        fps_text = f"FPS: {device.evsFps()}"
                        cv.putText(evs_image, fps_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv.imshow("EVS Camera", evs_image)
                        cv.waitKey(1)
                        break  # 只处理第一帧
                
                # 控制处理频率
                time.sleep(0.001)  # 1ms间隔
            
            # 清理窗口
            if device.evsModeIndex() > -1:
                cv.destroyWindow("EVS Camera")
            
        except Exception as e:
            logger.error(f"EVS显示线程错误: {e}")
    
    def _get_dummy_aps_frame(self) -> Optional[np.ndarray]:
        """获取模拟APS帧（用于测试或无数据时的显示）"""
        try:
            # 创建一个模拟的APS图像
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加APS相关信息
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            cv.putText(frame, f"APS - {timestamp}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if self.current_position:
                pos_text = f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})"
                cv.putText(frame, pos_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 添加数据采集状态
            if self.is_collecting_data:
                cv.putText(frame, "RECORDING", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 添加模拟的APS特征
            current_time = time.time()
            for i in range(5):
                x = int(100 + i * 100 + 50 * np.sin(current_time + i))
                y = int(240 + 50 * np.cos(current_time + i))
                cv.circle(frame, (x, y), 20, (0, 255, 255), -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"获取模拟APS帧失败: {e}")
            return None
    
    def _get_dummy_evs_frame(self) -> Optional[np.ndarray]:
        """获取模拟EVS帧（用于测试或无数据时的显示）"""
        try:
            # 创建一个模拟的EVS图像
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加EVS相关信息
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            cv.putText(frame, f"EVS - {timestamp}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if self.current_position:
                pos_text = f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})"
                cv.putText(frame, pos_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 添加数据采集状态
            if self.is_collecting_data:
                cv.putText(frame, "RECORDING", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 添加模拟的EVS特征（事件相机特点）
            current_time = time.time()
            for i in range(20):
                x = np.random.randint(0, 640)
                y = np.random.randint(0, 480)
                intensity = np.random.randint(0, 255)
                color = (0, intensity, intensity) if np.random.random() > 0.5 else (intensity, 0, 0)
                cv.circle(frame, (x, y), 2, color, -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"获取模拟EVS帧失败: {e}")
            return None
    
    def shutdown_camera(self):
        """完全关闭相机系统"""
        try:
            logger.info("开始关闭相机系统...")
            
            # 首先停止相机显示
            self.stop_camera_display()
            
            # 停止数据采集
            if self.is_collecting_data:
                self.stop_data_collection()
            
            # 关闭相机设备
            if hasattr(self.controller, 'camera') and self.controller.camera:
                device = self.controller.camera
                try:
                    if device.isOpened():
                        device.close()
                        logger.info("相机设备已关闭")
                    else:
                        logger.info("相机设备已经关闭")
                except Exception as e:
                    logger.error(f"关闭相机设备时出错: {e}")
                
                # 清理相机引用
                self.controller.camera = None
            
            logger.info("相机系统已完全关闭")
            
        except Exception as e:
            logger.error(f"关闭相机系统失败: {e}")
    
    def _get_dummy_hvs_frame(self) -> Optional[np.ndarray]:
        """获取模拟HVS帧（用于测试或无数据时的显示）"""
        try:
            # 创建一个模拟的HVS图像
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加HVS相关信息
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            cv.putText(frame, f"HVS - {timestamp}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if self.current_position:
                pos_text = f"Pos: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f})"
                cv.putText(frame, pos_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 添加数据采集状态
            if self.is_collecting_data:
                cv.putText(frame, "RECORDING", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 添加一些模拟的运动信息
            current_time = time.time()
            center_x = int(320 + 100 * np.sin(current_time * 2))
            center_y = int(240 + 100 * np.cos(current_time * 2))
            
            # 绘制模拟的运动轨迹
            cv.circle(frame, (center_x, center_y), 50, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"获取模拟HVS帧失败: {e}")
            return None
    
    def emergency_shutdown(self):
        """紧急关闭系统 - 在异常情况下确保所有资源被正确释放"""
        try:
            logger.warning("执行紧急关闭程序...")
            
            # 强制停止所有运行状态
            self.is_running = False
            self.display_running = False
            self.is_collecting_data = False
            
            # 紧急关闭相机系统
            self.shutdown_camera()
            
            # 紧急关闭机械臂
            if hasattr(self.controller, 'arm') and self.controller.arm:
                try:
                    self.controller.arm.disconnect()
                    logger.info("机械臂已断开连接")
                except Exception as e:
                    logger.error(f"断开机械臂连接时出错: {e}")
            
            # 强制清理所有线程
            for thread_name, thread in [
                ('APS显示线程', self.aps_display_thread),
                ('EVS显示线程', self.evs_display_thread),
                ('主相机显示线程', self.camera_display_thread)
            ]:
                if thread and thread.is_alive():
                    try:
                        logger.warning(f"强制终止{thread_name}")
                        # 注意：Python中没有直接强制终止线程的方法
                        # 这里只能设置标志位并记录警告
                    except Exception as e:
                        logger.error(f"终止{thread_name}时出错: {e}")
            
            # 清理线程引用
            self.aps_display_thread = None
            self.evs_display_thread = None
            self.camera_display_thread = None
            
            # 确保所有OpenCV窗口被关闭
            try:
                cv.destroyAllWindows()
            except Exception as e:
                logger.warning(f"关闭OpenCV窗口时出错: {e}")
            
            logger.info("紧急关闭程序执行完成")
            
        except Exception as e:
            logger.error(f"紧急关闭程序执行失败: {e}")
    
    def display_trajectory_menu(self) -> Optional[str]:
        """显示运动轨迹菜单"""
        print("\n" + "="*60)
        print("            运动轨迹选择菜单")
        print("="*60)
        
        for key, trajectory in self.trajectories.items():
            print(f"{key}. {trajectory['name']}")
            print(f"   {trajectory['description']}")
            print()
        
        print("q. 退出系统")
        print("="*60)
        
        while True:
            try:
                choice = input("请选择运动轨迹 (1-9): ").strip()
                if choice.lower() == 'q':
                    return None
                elif choice in self.trajectories:
                    return choice
                else:
                    print("无效选择，请输入 1-9 之间的数字或 q 退出")
            except KeyboardInterrupt:
                print("\n检测到中断信号")
                return None
            except Exception as e:
                print(f"输入错误: {e}")
    
    def start_data_collection(self) -> bool:
        """开始数据采集"""
        try:
            if self.is_collecting_data:
                logger.warning("数据采集已经在进行中")
                return True
            
            if not hasattr(self.controller, 'camera') or not self.controller.camera:
                logger.error("相机未初始化，无法开始数据采集")
                return False
            
            # 获取保存格式配置
            recording_config = self.controller.config.get_recording_config()
            save_format = recording_config.get('format', 'hdf5').lower()
            
            # 生成时间戳
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            milliseconds = datetime.now().microsecond // 1000
            timestamp = f"{current_time}{milliseconds:03d}"
            
            # 设置输出目录
            output_dir = recording_config.get('output_dir', './data')
            if recording_config.get('create_timestamp_dirs', True):
                output_dir = os.path.join(output_dir, timestamp)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 根据格式设置文件名和写入器类型
            if save_format == 'bin':
                file_name = os.path.join(output_dir, f"trajectory_data_{timestamp}")
                writer_type = WriterType.BIN
            else:  # hdf5
                file_name = os.path.join(output_dir, f"trajectory_data_{timestamp}.alpdata")
                writer_type = WriterType.HDF5
            
            # 获取设备属性
            device = self.controller.camera
            device_attribute = device.getDeviceAttribute(writer_type, file_name)
            
            # 获取APS和EVS属性
            aps_ptr = None
            evs_ptr = None
            
            if device.apsModeIndex() > -1:
                aps_ptr = device.getApsAttribute()
            
            if device.evsModeIndex() > -1:
                evs_ptr = device.getEvsAttribute()
            
            # 打开存储文件
            error = self.controller.writer_file.open(device_attribute, aps_ptr, evs_ptr)
            
            if error != AlpSaveFileError.NONE:
                logger.error(f"打开存储文件失败: {error}")
                return False
            
            self.is_collecting_data = True
            self.controller.current_output_dir = output_dir
            
            logger.info(f"数据采集已开始，保存格式: {save_format.upper()}")
            logger.info(f"输出目录: {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"开始数据采集失败: {e}")
            return False
    
    def start_data_collection_with_trajectory(self, trajectory: Dict[str, Any]) -> bool:
        """开始数据采集并记录轨迹信息"""
        try:
            if not self.start_data_collection():
                return False
            
            # 记录轨迹信息到文件
            self._save_trajectory_info(trajectory)
            
            return True
            
        except Exception as e:
            logger.error(f"开始数据采集并记录轨迹信息失败: {e}")
            return False
    
    def _prepare_trajectory_info(self, trajectory: Dict[str, Any]):
        """准备轨迹信息（在运动前创建文件和记录基本信息）"""
        try:
            # 获取保存格式配置
            recording_config = self.controller.config.get_recording_config()
            save_format = recording_config.get('format', 'hdf5').lower()
            
            # 生成时间戳
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            milliseconds = datetime.now().microsecond // 1000
            timestamp = f"{current_time}{milliseconds:03d}"
            
            # 设置输出目录
            output_dir = recording_config.get('output_dir', './data')
            if recording_config.get('create_timestamp_dirs', True):
                output_dir = os.path.join(output_dir, timestamp)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置输出目录供后续使用
            self.controller.current_output_dir = output_dir
            
            # 创建轨迹信息文件
            info_file = os.path.join(output_dir, "trajectory_info.txt")
            
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write("=== 运动轨迹信息 ===\n")
                f.write(f"轨迹名称: {trajectory['name']}\n")
                f.write(f"轨迹描述: {trajectory['description']}\n")
                f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                f.write(f"起始位置: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, z={self.current_position[2]:.2f}\n")
                
                # 记录运动参数
                params = trajectory['params']
                f.write("\n=== 运动参数 ===\n")
                for key, value in params.items():
                    if key == 'distance_mm':
                        f.write(f"移动距离: {value} 毫米\n")
                    elif key == 'angle_degrees':
                        f.write(f"旋转角度: {value} 度\n")
                    elif key == 'radius_mm':
                        f.write(f"圆形半径: {value} 毫米\n")
                    elif key == 'side_mm':
                        f.write(f"方形边长: {value} 毫米\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                # 记录系统配置
                f.write("\n=== 系统配置 ===\n")
                f.write(f"默认移动距离: {self.default_distance} 毫米\n")
                f.write(f"默认旋转角度: {self.default_rotation_angle} 度\n")
                f.write(f"线性移动速度: {self.linear_speed} mm/s\n")
                f.write(f"垂直移动速度: {self.vertical_speed} mm/s\n")
                f.write(f"旋转速度: {self.rotation_speed} °/s\n")
                
                # 记录相机配置
                if hasattr(self.controller, 'camera') and self.controller.camera:
                    device = self.controller.camera
                    f.write("\n=== 相机配置 ===\n")
                    f.write(f"相机模式: {self.controller.config.get('CAMERA', 'mode', fallback='HVS')}\n")
                    
                    if device.apsModeIndex() > -1:
                        f.write(f"APS分辨率: {device.apsWidth()}x{device.apsHeight()}\n")
                        f.write(f"APS帧率: {device.apsFps()} fps\n")
                        f.write(f"APS曝光时间: {device.apsExposureTime()} us\n")
                        f.write(f"APS增益: {device.apsAnalogGain()}\n")
                    
                    if device.evsModeIndex() > -1:
                        f.write(f"EVS分辨率: {device.evsWidth()}x{device.evsHeight()}\n")
                        f.write(f"EVS帧率: {device.evsFps()} fps\n")
                        f.write(f"EVS灵敏度: {device.evsSensitivity()}\n")
                
                f.write("\n=== 执行状态 ===\n")
                f.write("状态: 准备执行运动...\n")
            
            logger.info(f"轨迹信息已准备好: {info_file}")
            
        except Exception as e:
            logger.error(f"准备轨迹信息失败: {e}")
    
    def _save_trajectory_info(self, trajectory: Dict[str, Any]):
        """保存轨迹信息到文件"""
        try:
            if not hasattr(self.controller, 'current_output_dir') or not self.controller.current_output_dir:
                return
            
            # 创建轨迹信息文件
            info_file = os.path.join(self.controller.current_output_dir, "trajectory_info.txt")
            
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write("=== 运动轨迹信息 ===\n")
                f.write(f"轨迹名称: {trajectory['name']}\n")
                f.write(f"轨迹描述: {trajectory['description']}\n")
                f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                f.write(f"起始位置: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, z={self.current_position[2]:.2f}\n")
                
                # 记录运动参数
                params = trajectory['params']
                f.write("\n=== 运动参数 ===\n")
                for key, value in params.items():
                    if key == 'distance_mm':
                        f.write(f"移动距离: {value} 毫米\n")
                    elif key == 'angle_degrees':
                        f.write(f"旋转角度: {value} 度\n")
                    elif key == 'radius_mm':
                        f.write(f"圆形半径: {value} 毫米\n")
                    elif key == 'side_mm':
                        f.write(f"方形边长: {value} 毫米\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                # 记录系统配置
                f.write("\n=== 系统配置 ===\n")
                f.write(f"默认移动距离: {self.default_distance} 毫米\n")
                f.write(f"默认旋转角度: {self.default_rotation_angle} 度\n")
                
                # 记录相机配置
                if hasattr(self.controller, 'camera') and self.controller.camera:
                    device = self.controller.camera
                    f.write("\n=== 相机配置 ===\n")
                    f.write(f"相机模式: {self.controller.config.get('CAMERA', 'mode', fallback='HVS')}\n")
                    
                    if device.apsModeIndex() > -1:
                        f.write(f"APS分辨率: {device.apsWidth()}x{device.apsHeight()}\n")
                        f.write(f"APS帧率: {device.apsFps()} fps\n")
                        f.write(f"APS曝光时间: {device.apsExposureTime()} us\n")
                        f.write(f"APS增益: {device.apsAnalogGain()}\n")
                    
                    if device.evsModeIndex() > -1:
                        f.write(f"EVS分辨率: {device.evsWidth()}x{device.evsHeight()}\n")
                        f.write(f"EVS帧率: {device.evsFps()} fps\n")
                        f.write(f"EVS灵敏度: {device.evsSensitivity()}\n")
                
                f.write("\n=== 数据文件信息 ===\n")
                f.write(f"数据保存格式: {self.controller.config.get('RECORDING', 'format', fallback='hdf5')}\n")
                
                # 列出数据文件
                data_files = []
                for file in os.listdir(self.controller.current_output_dir):
                    if file.endswith(('.alpdata', '.bin', '.hdf5', '.h5')):
                        data_files.append(file)
                
                for data_file in data_files:
                    file_path = os.path.join(self.controller.current_output_dir, data_file)
                    file_size = os.path.getsize(file_path)
                    f.write(f"数据文件: {data_file} ({file_size} bytes)\n")
                
                f.write("\n=== 执行状态 ===\n")
                f.write("状态: 执行中...\n")
            
            logger.info(f"轨迹信息已保存到: {info_file}")
            
        except Exception as e:
            logger.error(f"保存轨迹信息失败: {e}")
    
    def _update_trajectory_completion_info(self, start_position: Optional[List[float]]):
        """更新轨迹完成信息"""
        try:
            if not hasattr(self.controller, 'current_output_dir') or not self.controller.current_output_dir:
                return
            
            # 更新轨迹信息文件
            info_file = os.path.join(self.controller.current_output_dir, "trajectory_info.txt")
            
            if not os.path.exists(info_file):
                return
            
            with open(info_file, 'a', encoding='utf-8') as f:
                f.write("\n=== 执行完成信息 ===\n")
                f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                
                if self.current_position:
                    f.write(f"结束位置: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, z={self.current_position[2]:.2f}\n")
                    
                    if start_position:
                        dx = self.current_position[0] - start_position[0]
                        dy = self.current_position[1] - start_position[1]
                        dz = self.current_position[2] - start_position[2]
                        f.write(f"位置变化: Δx={dx:.2f}mm, Δy={dy:.2f}mm, Δz={dz:.2f}mm\n")
                        
                        # 计算总移动距离
                        total_distance = np.sqrt(dx**2 + dy**2 + dz**2)
                        f.write(f"总移动距离: {total_distance:.2f}mm\n")
                
                f.write("状态: 执行完成\n")
                f.write("=" * 50 + "\n")
            
            logger.info(f"轨迹完成信息已更新到: {info_file}")
            
        except Exception as e:
            logger.error(f"更新轨迹完成信息失败: {e}")
    
    def stop_data_collection(self) -> bool:
        """停止数据采集"""
        try:
            if not self.is_collecting_data:
                logger.warning("数据采集未在进行中")
                return True
            
            if hasattr(self.controller, 'writer_file') and self.controller.writer_file:
                self.controller.writer_file.close()
            
            self.is_collecting_data = False
            logger.info("数据采集已停止")
            
            return True
            
        except Exception as e:
            logger.error(f"停止数据采集失败: {e}")
            return False
    
    def execute_trajectory(self, trajectory_key: str) -> bool:
        """执行选定的运动轨迹"""
        try:
            trajectory = self.trajectories[trajectory_key]
            logger.info(f"开始执行轨迹: {trajectory['name']}")
            
            # 记录起始位置
            start_position = self.current_position.copy() if self.current_position else None
            
            # 记录轨迹信息到文件（在运动前准备好）
            self._prepare_trajectory_info(trajectory)
            
            # 执行运动轨迹并在运动过程中同步采集数据
            result = trajectory['motion_func'](**trajectory['params'])
            
            # 更新当前位置
            self._update_current_position()
            
            # 停止数据采集并更新轨迹信息
            self.stop_data_collection()
            self._update_trajectory_completion_info(start_position)
            
            if result:
                logger.info(f"轨迹执行完成: {trajectory['name']}")
                print(f"\n✓ {trajectory['name']} 执行完成")
                if hasattr(self.controller, 'current_output_dir') and self.controller.current_output_dir:
                    print(f"  数据已保存到: {self.controller.current_output_dir}")
                
                if start_position and self.current_position:
                    dx = self.current_position[0] - start_position[0]
                    dy = self.current_position[1] - start_position[1]
                    dz = self.current_position[2] - start_position[2]
                    print(f"  位置变化: Δx={dx:.1f}mm, Δy={dy:.1f}mm, Δz={dz:.1f}mm")
                
                # 如果是回到初始位置，显示额外的信息
                if trajectory_key == '9' and self.initial_position:
                    print(f"  已回到初始位置: x={self.initial_position[0]:.1f}mm, y={self.initial_position[1]:.1f}mm, z={self.initial_position[2]:.1f}mm")
                
                return True
            else:
                logger.error(f"轨迹执行失败: {trajectory['name']}")
                print(f"\n✗ {trajectory['name']} 执行失败")
                return False
                
        except Exception as e:
            logger.error(f"执行轨迹时发生错误: {e}")
            print(f"\n✗ 执行过程中发生错误: {e}")
            return False
    
    def _update_current_position(self):
        """更新当前位置信息"""
        try:
            position_data = self.controller.arm.get_position()
            if isinstance(position_data, tuple) and len(position_data) == 2:
                actual_position = position_data[1]
            elif isinstance(position_data, (list, tuple)) and len(position_data) >= 6:
                actual_position = position_data
            else:
                logger.error(f"位置数据格式不正确: {position_data}")
                return
            
            if isinstance(actual_position, (list, tuple)) and len(actual_position) >= 6:
                self.current_position = list(actual_position)[:6]
                logger.info(f"位置已更新: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, z={self.current_position[2]:.2f}")
            else:
                logger.error(f"实际位置数据格式不正确: {actual_position}")
                
        except Exception as e:
            logger.error(f"更新当前位置失败: {e}")
    
    # 运动轨迹实现方法
    def _move_horizontal_right(self, distance_mm: int) -> bool:
        """水平向右移动"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行相对移动")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标位置（基于当前位置的相对移动）
        target_x = self.current_position[0] + distance_mm
        target_y = self.current_position[1]
        target_z = self.current_position[2]
        result = self.controller.move_linear(x=target_x, y=target_y, z=target_z, speed=self.linear_speed)
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _move_horizontal_left(self, distance_mm: int) -> bool:
        """水平向左移动"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行相对移动")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标位置（基于当前位置的相对移动）
        target_x = self.current_position[0] - distance_mm
        target_y = self.current_position[1]
        target_z = self.current_position[2]
        result = self.controller.move_linear(x=target_x, y=target_y, z=target_z, speed=self.linear_speed)
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _move_vertical_up(self, distance_mm: int) -> bool:
        """垂直向上移动"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行相对移动")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标位置（基于当前位置的相对移动）
        target_x = self.current_position[0]
        target_y = self.current_position[1]
        target_z = self.current_position[2] + distance_mm
        result = self.controller.move_linear(x=target_x, y=target_y, z=target_z, speed=self.vertical_speed)
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _move_vertical_down(self, distance_mm: int) -> bool:
        """垂直向下移动"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行相对移动")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标位置（基于当前位置的相对移动）
        target_x = self.current_position[0]
        target_y = self.current_position[1]
        target_z = self.current_position[2] - distance_mm
        result = self.controller.move_linear(x=target_x, y=target_y, z=target_z, speed=self.vertical_speed)
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _move_y_positive(self, distance_mm: int) -> bool:
        """Y+方向移动"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行相对移动")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标位置（基于当前位置的相对移动）
        target_x = self.current_position[0]
        target_y = self.current_position[1] + distance_mm
        target_z = self.current_position[2]
        result = self.controller.move_linear(x=target_x, y=target_y, z=target_z, speed=self.linear_speed)
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _move_y_negative(self, distance_mm: int) -> bool:
        """Y-方向移动"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行相对移动")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标位置（基于当前位置的相对移动）
        target_x = self.current_position[0]
        target_y = self.current_position[1] - distance_mm
        target_z = self.current_position[2]
        result = self.controller.move_linear(x=target_x, y=target_y, z=target_z, speed=self.linear_speed)
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _rotate_counter_clockwise(self, angle_degrees: int) -> bool:
        """逆时针旋转"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行旋转")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标角度（基于当前角度的相对旋转）
        target_yaw = self.current_position[5] + angle_degrees
        
        # 处理可能的inf值
        roll = self.current_position[3] if not np.isinf(self.current_position[3]) else 0
        pitch = self.current_position[4] if not np.isinf(self.current_position[4]) else 0
        
        result = self.controller.move_linear(
            x=self.current_position[0], 
            y=self.current_position[1], 
            z=self.current_position[2],
            roll=roll,
            pitch=pitch,
            yaw=target_yaw, 
            speed=self.rotation_speed
        )
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    def _rotate_clockwise(self, angle_degrees: int) -> bool:
        """顺时针旋转"""
        if not self.current_position:
            logger.error("当前位置未知，无法执行旋转")
            return False
        
        # 开始同步数据采集
        if not self._start_motion_data_collection():
            logger.warning("数据采集启动失败，继续执行运动")
        
        # 计算目标角度（基于当前角度的相对旋转）
        target_yaw = self.current_position[5] - angle_degrees
        
        # 处理可能的inf值
        roll = self.current_position[3] if not np.isinf(self.current_position[3]) else 0
        pitch = self.current_position[4] if not np.isinf(self.current_position[4]) else 0
        
        result = self.controller.move_linear(
            x=self.current_position[0], 
            y=self.current_position[1], 
            z=self.current_position[2],
            roll=roll,
            pitch=pitch,
            yaw=target_yaw, 
            speed=self.rotation_speed
        )
        
        # 运动完成后停止数据采集
        self._stop_motion_data_collection()
        
        return result
    
    
    def _move_to_initial_position(self) -> bool:
        """回到初始位置"""
        try:
            logger.info("开始回到初始位置")
            
            if not self.initial_position:
                logger.error("初始位置未保存，无法返回初始位置")
                return False
            
            logger.info(f"目标初始位置: x={self.initial_position[0]:.2f}, y={self.initial_position[1]:.2f}, z={self.initial_position[2]:.2f}")
            
            # 开始同步数据采集
            if not self._start_motion_data_collection():
                logger.warning("数据采集启动失败，继续执行运动")
            
            # 移动到初始位置
            result = self.controller.move_linear(
                x=self.initial_position[0],
                y=self.initial_position[1],
                z=self.initial_position[2],
                roll=self.initial_position[3],
                pitch=self.initial_position[4],
                yaw=self.initial_position[5],
                speed=self.linear_speed
            )
            
            # 运动完成后停止数据采集
            self._stop_motion_data_collection()
            
            if result:
                logger.info("已成功回到初始位置")
                return True
            else:
                logger.error("回到初始位置失败")
                return False
                
        except Exception as e:
            logger.error(f"回到初始位置失败: {e}")
            # 异常情况下也要停止数据采集
            self._stop_motion_data_collection()
            return False
    
    def run(self):
        """运行运动轨迹控制系统"""
        try:
            # 确认启动
            if not self.confirm_startup():
                print("系统启动已取消")
                return
            
            # 初始化系统
            if not self.initialize_systems():
                print("系统初始化失败，程序退出")
                return
            
            # 启动相机显示
            self.start_camera_display()
            
            print("\n" + "="*60)
            print("          系统已就绪，可以开始运动控制")
            print("="*60)
            print("提示：")
            print("• APS和EVS相机窗口中按 'q' 键可关闭显示")
            print("• 执行运动轨迹时自动开始数据采集")
            print("• 数据保存格式由settings.conf配置决定")
            print(f"• 默认移动距离: {self.default_distance}毫米")
            print(f"• 默认旋转角度: {self.default_rotation_angle}度")
            print()
            
            self.is_running = True
            
            # 主循环
            while self.is_running:
                try:
                    # 显示轨迹选择菜单
                    trajectory_key = self.display_trajectory_menu()
                    
                    if trajectory_key is None:
                        break
                    
                    # 执行选定的轨迹
                    self.execute_trajectory(trajectory_key)
                    
                    # 询问是否继续
                    while True:
                        continue_choice = input("\n是否继续选择其他轨迹？ (y/n): ").strip().lower()
                        if continue_choice in ['y', 'yes', '是']:
                            break
                        elif continue_choice in ['n', 'no', '否']:
                            self.is_running = False
                            break
                        else:
                            print("请输入 y/yes/是 或 n/no/否")
                    
                except KeyboardInterrupt:
                    print("\n检测到中断信号")
                    break
                except Exception as e:
                    logger.error(f"主循环错误: {e}")
                    print(f"发生错误: {e}")
                    break
            
            print("\n正在关闭系统...")
            
        except Exception as e:
            logger.error(f"系统运行错误: {e}")
            print(f"系统运行错误: {e}")
        
        finally:
            # 清理资源
            logger.info("开始系统清理流程...")
            
            try:
                # 停止所有运行状态
                self.is_running = False
                self.display_running = False
                
                # 停止数据采集
                if self.is_collecting_data:
                    self.stop_data_collection()
                
                # 完全关闭相机系统
                self.shutdown_camera()
                
                # 清理控制器资源
                if hasattr(self.controller, 'cleanup'):
                    self.controller.cleanup()
                
                # 确保所有OpenCV窗口被关闭
                try:
                    cv.destroyAllWindows()
                except Exception as e:
                    logger.warning(f"关闭OpenCV窗口时出错: {e}")
                
                # 清理线程引用
                self.aps_display_thread = None
                self.evs_display_thread = None
                self.camera_display_thread = None
                
                logger.info("系统资源清理完成")
                
            except Exception as e:
                logger.error(f"系统清理过程中出错: {e}")
                # 即使出错也要尝试紧急关闭
                try:
                    self.emergency_shutdown()
                except Exception as emergency_e:
                    logger.error(f"紧急关闭也失败: {emergency_e}")
            
            print("系统已安全关闭")
            logger.info("运动轨迹控制系统已安全关闭")


def main():
    """主函数"""
    print("EVA_CAM 运动轨迹控制系统")
    print("按 Ctrl+C 可随时退出程序")
    
    system = None
    try:
        system = TrajectoryControlSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        logger.info("程序被用户中断")
        # 确保系统被正确关闭
        if system:
            try:
                system.emergency_shutdown()
            except Exception as e:
                logger.error(f"紧急关闭失败: {e}")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        print(f"程序执行错误: {e}")
        # 确保系统被正确关闭
        if system:
            try:
                system.emergency_shutdown()
            except Exception as emergency_e:
                logger.error(f"紧急关闭失败: {emergency_e}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()