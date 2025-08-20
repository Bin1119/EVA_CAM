#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式相机控制脚本
打开相机并等待按键指令控制机械臂水平移动
"""

import sys
import os
import time
import threading
import keyboard
import configparser
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_cam_controller import EvaCamController
from utils import logger


class InteractiveCameraControl:
    """交互式相机控制类"""
    
    def __init__(self):
        self.controller = EvaCamController()
        self.is_running = False
        self.current_position = None
        self.config = configparser.ConfigParser()
        self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.conf')
        try:
            self.config.read(config_path)
            logger.info(f"配置文件加载成功: {config_path}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            
    def get_config_value(self, section: str, key: str, fallback=None):
        """获取配置值"""
        try:
            return self.config.get(section, key, fallback=fallback)
        except:
            return fallback
        
    def initialize_systems(self, no_home: bool = True) -> bool:
        """
        初始化系统
        
        Args:
            no_home: 是否跳过归零动作
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=== 初始化交互式相机控制系统 ===")
            
            # 初始化机械臂
            if not self.controller.initialize_xarm(go_home=not no_home):
                logger.error("机械臂初始化失败")
                return False
            
            # 初始化相机
            if not self.controller.initialize_camera():
                logger.error("相机初始化失败")
                return False
            
            # 获取当前位置
            position_data = self.controller.arm.get_position()
            logger.info(f"获取到的位置数据: {position_data}")
            logger.info(f"位置数据类型: {type(position_data)}")
            
            # 处理不同的位置数据格式
            if isinstance(position_data, tuple) and len(position_data) == 2:
                # 格式: (0, [x, y, z, roll, pitch, yaw])
                actual_position = position_data[1]
            elif isinstance(position_data, (list, tuple)) and len(position_data) >= 6:
                # 格式: [x, y, z, roll, pitch, yaw, ...]
                actual_position = position_data
            else:
                logger.error(f"位置数据格式不正确: {position_data}")
                return False
            
            # 确保实际位置数据是列表格式
            if isinstance(actual_position, (list, tuple)) and len(actual_position) >= 6:
                self.current_position = list(actual_position)[:6]  # 取前6个元素 [x, y, z, roll, pitch, yaw]
            else:
                logger.error(f"实际位置数据格式不正确: {actual_position}")
                return False
            
            logger.info(f"当前位置: x={self.current_position[0]}, y={self.current_position[1]}, z={self.current_position[2]}")
            
            logger.info("系统初始化完成！")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def move_horizontal_right(self, distance_mm: int = 10) -> bool:
        """
        机械臂水平向右移动
        
        Args:
            distance_mm: 移动距离（毫米）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_position:
                logger.error("无法获取当前位置")
                return False
            
            # 计算目标位置（Y轴正方向为右）
            target_position = self.current_position.copy()
            target_position[1] = target_position[1] + distance_mm  # Y轴正方向
            
            target_x = target_position[0]
            target_y = target_position[1]
            target_z = target_position[2]
            target_roll = target_position[3]
            target_pitch = target_position[4]
            target_yaw = target_position[5]
            
            logger.info(f"向右移动 {distance_mm}mm")
            logger.info(f"目标位置: x={target_x}, y={target_y}, z={target_z}")
            
            # 执行移动
            success = self.controller.execute_motion_with_data_collection(
                self.controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=self.get_config_value('MOTION', 'linear_speed', 100), wait=True
            )
            
            if success:
                # 更新当前位置
                self.current_position = target_position.copy()
                logger.info("移动完成！")
            else:
                logger.error("移动失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"移动过程中出错: {e}")
            return False
    
    def move_horizontal_left(self, distance_mm: int = 10) -> bool:
        """
        机械臂水平向左移动
        
        Args:
            distance_mm: 移动距离（毫米）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_position:
                logger.error("无法获取当前位置")
                return False
            
            # 计算目标位置（Y轴负方向为左）
            target_position = self.current_position.copy()
            target_position[1] = target_position[1] - distance_mm  # Y轴负方向
            
            target_x = target_position[0]
            target_y = target_position[1]
            target_z = target_position[2]
            target_roll = target_position[3]
            target_pitch = target_position[4]
            target_yaw = target_position[5]
            
            logger.info(f"向左移动 {distance_mm}mm")
            logger.info(f"目标位置: x={target_x}, y={target_y}, z={target_z}")
            
            # 执行移动
            success = self.controller.execute_motion_with_data_collection(
                self.controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=self.get_config_value('MOTION', 'linear_speed', 100), wait=True
            )
            
            if success:
                # 更新当前位置
                self.current_position = target_position.copy()
                logger.info("移动完成！")
            else:
                logger.error("移动失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"移动过程中出错: {e}")
            return False
    
    def move_vertical_up(self, distance_mm: int = 10) -> bool:
        """
        机械臂垂直向上移动
        
        Args:
            distance_mm: 移动距离（毫米）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_position:
                logger.error("无法获取当前位置")
                return False
            
            # 计算目标位置（Z轴正方向为上）
            target_position = self.current_position.copy()
            target_position[2] = target_position[2] + distance_mm  # Z轴正方向
            
            target_x = target_position[0]
            target_y = target_position[1]
            target_z = target_position[2]
            target_roll = target_position[3]
            target_pitch = target_position[4]
            target_yaw = target_position[5]
            
            logger.info(f"向上移动 {distance_mm}mm")
            logger.info(f"目标位置: x={target_x}, y={target_y}, z={target_z}")
            
            # 执行移动
            success = self.controller.execute_motion_with_data_collection(
                self.controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=self.get_config_value('MOTION', 'vertical_speed', 100), wait=True
            )
            
            if success:
                # 更新当前位置
                self.current_position = target_position.copy()
                logger.info("移动完成！")
            else:
                logger.error("移动失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"移动过程中出错: {e}")
            return False
    
    def move_vertical_down(self, distance_mm: int = 10) -> bool:
        """
        机械臂垂直向下移动
        
        Args:
            distance_mm: 移动距离（毫米）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_position:
                logger.error("无法获取当前位置")
                return False
            
            # 计算目标位置（Z轴负方向为下）
            target_position = self.current_position.copy()
            target_position[2] = target_position[2] - distance_mm  # Z轴负方向
            
            target_x = target_position[0]
            target_y = target_position[1]
            target_z = target_position[2]
            target_roll = target_position[3]
            target_pitch = target_position[4]
            target_yaw = target_position[5]
            
            logger.info(f"向下移动 {distance_mm}mm")
            logger.info(f"目标位置: x={target_x}, y={target_y}, z={target_z}")
            
            # 执行移动
            success = self.controller.execute_motion_with_data_collection(
                self.controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=self.get_config_value('MOTION', 'vertical_speed', 100), wait=True
            )
            
            if success:
                # 更新当前位置
                self.current_position = target_position.copy()
                logger.info("移动完成！")
            else:
                logger.error("移动失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"移动过程中出错: {e}")
            return False
    
    def rotate_counter_clockwise(self, angle_deg: int = 1) -> bool:
        """
        机械臂水平逆时针旋转
        
        Args:
            angle_deg: 旋转角度（度）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_position:
                logger.error("无法获取当前位置")
                return False
            
            # 计算目标角度（yaw增加为逆时针）
            target_position = self.current_position.copy()
            target_position[5] = target_position[5] + angle_deg  # yaw轴增加
            
            target_x = target_position[0]
            target_y = target_position[1]
            target_z = target_position[2]
            target_roll = target_position[3]
            target_pitch = target_position[4]
            target_yaw = target_position[5]
            
            logger.info(f"逆时针旋转 {angle_deg}°")
            logger.info(f"目标角度: yaw={target_yaw}°")
            
            # 执行旋转
            success = self.controller.execute_motion_with_data_collection(
                self.controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=self.get_config_value('MOTION', 'rotation_speed', 50), wait=True
            )
            
            if success:
                # 更新当前位置
                self.current_position = target_position.copy()
                logger.info("旋转完成！")
            else:
                logger.error("旋转失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"旋转过程中出错: {e}")
            return False
    
    def rotate_clockwise(self, angle_deg: int = 1) -> bool:
        """
        机械臂水平顺时针旋转
        
        Args:
            angle_deg: 旋转角度（度）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_position:
                logger.error("无法获取当前位置")
                return False
            
            # 计算目标角度（yaw减少为顺时针）
            target_position = self.current_position.copy()
            target_position[5] = target_position[5] - angle_deg  # yaw轴减少
            
            target_x = target_position[0]
            target_y = target_position[1]
            target_z = target_position[2]
            target_roll = target_position[3]
            target_pitch = target_position[4]
            target_yaw = target_position[5]
            
            logger.info(f"顺时针旋转 {angle_deg}°")
            logger.info(f"目标角度: yaw={target_yaw}°")
            
            # 执行旋转
            success = self.controller.execute_motion_with_data_collection(
                self.controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=self.get_config_value('MOTION', 'rotation_speed', 50), wait=True
            )
            
            if success:
                # 更新当前位置
                self.current_position = target_position.copy()
                logger.info("旋转完成！")
            else:
                logger.error("旋转失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"旋转过程中出错: {e}")
            return False
    
    def keyboard_listener(self):
        """键盘监听线程"""
        logger.info("按键监听已启动")
        logger.info("控制说明：")
        logger.info("按住 'A' 键：持续水平向右移动")
        logger.info("按住 'D' 键：持续水平向左移动")
        logger.info("按住 'W' 键：持续垂直向上移动")
        logger.info("按住 'S' 键：持续垂直向下移动")
        logger.info("按住 'Q' 键：持续水平逆时针旋转")
        logger.info("按住 'E' 键：持续水平顺时针旋转")
        logger.info("按 'ESC' 键：退出程序")
        
        # 移动参数
        move_step = 5  # 每次移动1毫米（更精细的控制）
        rotate_step = 2  # 每次旋转0.5度
        move_interval = 0.001  # 移动间隔20毫秒（50Hz，更流畅）
        
        # 记录按键状态
        key_states = {
            'a': False,
            'd': False, 
            'w': False,
            's': False,
            'q': False,
            'e': False
        }
        
        while self.is_running:
            try:
                # 检测按键状态变化
                current_a = keyboard.is_pressed('a')
                current_d = keyboard.is_pressed('d')
                current_w = keyboard.is_pressed('w')
                current_s = keyboard.is_pressed('s')
                current_q = keyboard.is_pressed('q')
                current_e = keyboard.is_pressed('e')
                current_esc = keyboard.is_pressed('esc')
                
                # 检测新按下的键
                if current_a and not key_states['a']:
                    logger.info("开始水平向右移动")
                    key_states['a'] = True
                elif not current_a and key_states['a']:
                    logger.info("停止水平向右移动")
                    key_states['a'] = False
                    
                if current_d and not key_states['d']:
                    logger.info("开始水平向左移动")
                    key_states['d'] = True
                elif not current_d and key_states['d']:
                    logger.info("停止水平向左移动")
                    key_states['d'] = False
                    
                if current_w and not key_states['w']:
                    logger.info("开始垂直向上移动")
                    key_states['w'] = True
                elif not current_w and key_states['w']:
                    logger.info("停止垂直向上移动")
                    key_states['w'] = False
                    
                if current_s and not key_states['s']:
                    logger.info("开始垂直向下移动")
                    key_states['s'] = True
                elif not current_s and key_states['s']:
                    logger.info("停止垂直向下移动")
                    key_states['s'] = False
                    
                if current_q and not key_states['q']:
                    logger.info("开始水平逆时针旋转")
                    key_states['q'] = True
                elif not current_q and key_states['q']:
                    logger.info("停止水平逆时针旋转")
                    key_states['q'] = False
                    
                if current_e and not key_states['e']:
                    logger.info("开始水平顺时针旋转")
                    key_states['e'] = True
                elif not current_e and key_states['e']:
                    logger.info("停止水平顺时针旋转")
                    key_states['e'] = False
                
                # 执行连续移动
                if key_states['a']:
                    self.move_horizontal_right(move_step)
                elif key_states['d']:
                    self.move_horizontal_left(move_step)
                elif key_states['w']:
                    self.move_vertical_up(move_step)
                elif key_states['s']:
                    self.move_vertical_down(move_step)
                elif key_states['q']:
                    self.rotate_counter_clockwise(rotate_step)
                elif key_states['e']:
                    self.rotate_clockwise(rotate_step)
                
                # 检测退出键
                if current_esc:
                    logger.info("检测到 'ESC' 键按下，准备退出")
                    self.is_running = False
                    break
                
                time.sleep(move_interval)
                
            except Exception as e:
                logger.error(f"键盘监听错误: {e}")
                break
    
    def run(self):
        """运行交互式控制"""
        try:
            # 初始化系统
            if not self.initialize_systems(no_home=True):
                logger.error("系统初始化失败，程序退出")
                return
            
            # 启动键盘监听线程
            self.is_running = True
            listener_thread = threading.Thread(target=self.keyboard_listener)
            listener_thread.daemon = True
            listener_thread.start()
            
            logger.info("=== 交互式控制已启动 ===")
            logger.info("相机正在运行，等待按键指令...")
            
            # 主线程等待
            while self.is_running:
                time.sleep(0.1)
            
            logger.info("正在关闭系统...")
            
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
        except Exception as e:
            logger.error(f"程序运行错误: {e}")
        finally:
            self.is_running = False
            self.controller.disconnect()
            logger.info("程序已退出")


def main():
    """主函数"""
    print("=== 交互式相机控制系统 ===")
    print("功能说明：")
    print("1. 读取配置文件并初始化机械臂和相机")
    print("2. 使用键盘控制机械臂（按住按键可持续移动）：")
    print("   - 按住A键：持续水平向右移动")
    print("   - 按住D键：持续水平向左移动")
    print("   - 按住W键：持续垂直向上移动")
    print("   - 按住S键：持续垂直向下移动")
    print("   - 按住Q键：持续水平逆时针旋转")
    print("   - 按住E键：持续水平顺时针旋转")
    print("   - 按ESC键：退出程序")
    print("3. 移动参数：每步1毫米，旋转每步0.5度，间隔20毫秒")
    print("4. 按 Ctrl+C 也可以退出程序")
    print()
    
    # 确认是否继续
    try:
        confirm = input("确认启动系统? (y/n): ").strip().lower()
        if confirm != 'y':
            print("程序已取消")
            return
    except KeyboardInterrupt:
        print("\n程序已取消")
        return
    
    # 创建控制器并运行
    controller = InteractiveCameraControl()
    controller.run()


if __name__ == "__main__":
    main()