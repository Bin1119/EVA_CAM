# -*- coding: utf-8 -*-

import os
import sys

# 查询目录地址
current_dir = os.path.dirname(os.path.abspath(__file__))
# 查询pyd地址
py3_path = os.path.join(current_dir, "..\\bin")
# 查询到AlpPython
sys.path.append(py3_path)

# 导入提供的python库
from AlpPython import *
import numpy as np
import cv2 as cv
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

def extract_timestamp_from_data(data_obj) -> Optional[int]:
    """
    从数据对象中提取时间戳
    
    :param data_obj: 数据对象（APS或EVS）
    :return: 时间戳（微秒）或None
    """
    for method_name in ['getTimestamp', 'timestamp', 'getTime', 'time']:
        try:
            timestamp_us = getattr(data_obj, method_name)()
            return int(timestamp_us)
        except:
            continue
    return None

def extract_aps_info(aps_data, frame_number: int) -> Optional[Dict[str, Any]]:
    """
    提取APS数据信息
    
    :param aps_data: APS数据对象
    :param frame_number: 帧号
    :return: APS数据信息字典或None
    """
    try:
        # 转换为OpenCV格式
        aps_image = aps_data.convertTo()
        
        # 获取时间戳
        timestamp_us = extract_timestamp_from_data(aps_data)
        if timestamp_us is None:
            return None
        
        # 获取数据尺寸
        if hasattr(aps_image, 'shape'):
            height, width = aps_image.shape[:2]
            size_str = f"{width}x{height}"
        else:
            size_str = "N/A"
        
        return {
            'frame_number': frame_number,
            'data_type': 'APS',
            'timestamp_us': timestamp_us,
            'size_str': size_str
        }
    except Exception as e:
        return None

def extract_evs_info(evs_data, frame_number: int) -> Optional[Dict[str, Any]]:
    """
    提取EVS数据信息
    
    :param evs_data: EVS数据对象
    :param frame_number: 帧号
    :return: EVS数据信息字典或None
    """
    try:
        # 获取时间戳
        timestamp_us = extract_timestamp_from_data(evs_data)
        if timestamp_us is None:
            return None
        
        # 尝试获取事件数量和尺寸信息
        event_count = 0
        size_str = "N/A"
        
        # 尝试获取事件数量
        if hasattr(evs_data, 'getEventCount'):
            event_count = evs_data.getEventCount()
        elif hasattr(evs_data, 'event_count'):
            event_count = evs_data.event_count
        elif hasattr(evs_data, 'size'):
            event_count = evs_data.size()
        
        # 尝试获取传感器尺寸
        if hasattr(evs_data, 'getWidth') and hasattr(evs_data, 'getHeight'):
            try:
                width = evs_data.getWidth()
                height = evs_data.getHeight()
                if callable(width):
                    width = width()
                if callable(height):
                    height = height()
                size_str = f"{width}x{height}"
            except:
                size_str = "N/A"
        elif hasattr(evs_data, 'width') and hasattr(evs_data, 'height'):
            try:
                width = evs_data.width
                height = evs_data.height
                if callable(width):
                    width = width()
                if callable(height):
                    height = height()
                size_str = f"{width}x{height}"
            except:
                size_str = "N/A"
        else:
            # 如果无法获取事件信息，尝试从渲染帧获取尺寸
            try:
                evs_image = evs_data.frame()
                if hasattr(evs_image, 'shape'):
                    height, width = evs_image.shape[:2]
                    size_str = f"{width}x{height}"
            except:
                size_str = "N/A"
        
        # 根据是否有事件数量决定数据类型
        if event_count > 0:
            return {
                'frame_number': frame_number,
                'data_type': 'EVS_Events',
                'timestamp_us': timestamp_us,
                'size_str': f"{size_str}\t事件数:{event_count}"
            }
        else:
            return {
                'frame_number': frame_number,
                'data_type': 'EVS',
                'timestamp_us': timestamp_us,
                'size_str': size_str
            }
    except Exception as e:
        return None

def extractMotionData(player, output_file_path, data_file_path, sync_timestamps_path):
    """
    提取机械臂运动期间的APS和EVS数据及其时间戳
    
    :param player: 播放器实例
    :param output_file_path: 输出txt文件路径
    :param data_file_path: 数据文件路径
    :param sync_timestamps_path: 同步时间戳文件路径
    """
    
    # 用于存储第一个配对的APS和EVS数据
    first_pair_aps = None
    first_pair_evs = None
    
    # 读取同步时间戳文件
    motion_start_time = None
    motion_end_time = None
    pre_motion_time = None
    
    try:
        with open(sync_timestamps_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if "预运动时间:" in line:
                    pre_motion_time = float(line.split("预运动时间: ")[1].strip())
                elif "运动开始时间:" in line:
                    motion_start_time = float(line.split("运动开始时间: ")[1].strip())
                elif "运动结束时间:" in line:
                    motion_end_time = float(line.split("运动结束时间: ")[1].strip())
    except Exception as e:
        print(f"读取同步时间戳文件失败: {e}")
        return
    
    if motion_start_time is None or motion_end_time is None:
        print("无法获取运动时间范围")
        return
    
    print(f"运动时间范围: {motion_start_time} - {motion_end_time}")
    print(f"运动持续时间: {motion_end_time - motion_start_time} 秒")
    
    # 初始化变量
    frame_count = 0
    motion_aps_count = 0
    motion_evs_count = 0
    aps_timestamps = []
    evs_timestamps = []
    all_data = []
    
    # 存储原始数据对象
    raw_aps_data = {}  # 帧号 -> APS数据对象
    raw_evs_data = {}  # 帧号 -> EVS数据对象
    
    # 创建输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("=== 机械臂运动期间数据提取结果 ===\n")
        f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件路径: {data_file_path}\n")
        f.write(f"同步时间戳文件: {sync_timestamps_path}\n")
        f.write(f"运动开始时间: {motion_start_time}\n")
        f.write(f"运动结束时间: {motion_end_time}\n")
        f.write(f"运动持续时间: {motion_end_time - motion_start_time} 秒\n")
        f.write("-" * 90 + "\n")
        f.write("帧号\t数据类型\t时间戳(μs)\t\t系统时间戳\t\t\t是否在运动期间\t数据尺寸/事件信息\n")
        f.write("-" * 90 + "\n")
        
        print("正在收集所有数据...")
        
        # 第一次遍历：收集所有APS和EVS数据
        while player.isWorking():
            sync_list = player.getSyncFrames()
            
            for it in sync_list:
                # 获取APS数据信息
                if len(it) > 0 and it[0] is not None:
                    frame_count += 1
                    aps_info = extract_aps_info(it[0], frame_count)
                    if aps_info:
                        all_data.append(aps_info)
                        raw_aps_data[frame_count] = it[0]  # 存储原始APS数据
                        if frame_count % 10 == 0:
                            print(f"已收集 {frame_count} 帧数据...")
                
                # 获取EVS数据信息
                if len(it) > 1 and len(it[1]) > 0:
                    for evs_data in it[1]:
                        frame_count += 1
                        evs_info = extract_evs_info(evs_data, frame_count)
                        if evs_info:
                            all_data.append(evs_info)
                            raw_evs_data[frame_count] = evs_data  # 存储原始EVS数据
            
            # 避免过度占用CPU
            time.sleep(0.001)
        
        print(f"数据收集完成，总共收集了 {len(all_data)} 帧数据")
        
        if not all_data:
            print("未收集到任何数据")
            return
        
        # 按时间戳排序数据
        all_data.sort(key=lambda x: x['timestamp_us'])
        print(f"数据已按时间戳排序")
        
        # 找到第一个APS数据作为基准
        first_aps_data = None
        for data in all_data:
            if data['data_type'] == 'APS':
                first_aps_data = data
                break
        
        if first_aps_data is None:
            print("未找到APS数据作为基准")
            return
        
        # 计算相对时间戳来选取运动期间的数据
        first_aps_timestamp = first_aps_data['timestamp_us']  # 5286019 μs
        pre_motion_time_abs = pre_motion_time  # 1756032262.636577
        
        # 计算运动开始和结束相对于预运动时间的偏移
        motion_start_offset = motion_start_time - pre_motion_time_abs  # 运动开始相对于预运动时间的偏移
        motion_end_offset = motion_end_time - pre_motion_time_abs      # 运动结束相对于预运动时间的偏移
        
        print(f"预运动时间: {pre_motion_time_abs}")
        print(f"第一个APS时间戳: {first_aps_timestamp} μs")
        print(f"运动开始相对偏移: {motion_start_offset} 秒")
        print(f"运动结束相对偏移: {motion_end_offset} 秒")
        print(f"运动总持续时间: {motion_end_offset - motion_start_offset} 秒")
        
        # 计算运动期间对应的时间戳范围
        motion_start_timestamp = first_aps_timestamp + int(motion_start_offset * 1000000)
        motion_end_timestamp = first_aps_timestamp + int(motion_end_offset * 1000000)
        
        print(f"运动期间时间戳范围: {motion_start_timestamp} - {motion_end_timestamp} μs")
        
        # 第二次遍历：分析哪些数据在运动期间
        print("正在分析运动期间的数据...")
        
        for data_info in all_data:
            # 检查时间戳是否在运动期间范围内
            is_in_motion = motion_start_timestamp <= data_info['timestamp_us'] <= motion_end_timestamp
            
            # 计算对应的系统时间戳（用于显示）
            system_time = pre_motion_time_abs + (data_info['timestamp_us'] - first_aps_timestamp) / 1000000.0
            
            if is_in_motion:
                if data_info['data_type'] == 'APS':
                    motion_aps_count += 1
                    aps_timestamps.append({
                        'frame_number': data_info['frame_number'],
                        'timestamp_us': data_info['timestamp_us'],
                        'system_time': system_time,
                        'is_in_motion': True
                    })
                elif 'EVS' in data_info['data_type']:
                    motion_evs_count += 1
                    evs_timestamps.append({
                        'frame_number': data_info['frame_number'],
                        'timestamp_us': data_info['timestamp_us'],
                        'system_time': system_time,
                        'is_in_motion': True,
                        'data_type': data_info['data_type']
                    })
            
            # 写入文件
            motion_status = "是" if is_in_motion else "否"
            f.write(f"{data_info['frame_number']}\t{data_info['data_type']}\t{data_info['timestamp_us']}\t\t{system_time:.6f}\t{motion_status}\t\t{data_info['size_str']}\n")
            f.flush()
            
            # 打印运动期间的数据
            if is_in_motion:
                print(f"发现运动期间{data_info['data_type']}数据 - 帧号: {data_info['frame_number']}, 时间戳: {data_info['timestamp_us']} μs, 系统时间: {system_time:.6f}")
        
        f.write("-" * 90 + "\n")
        f.write(f"处理完成，总共处理了 {frame_count} 帧数据\n")
        f.write(f"运动期间APS数据数量: {motion_aps_count}\n")
        f.write(f"运动期间EVS数据数量: {motion_evs_count}\n")
        f.write(f"运动期间数据总数: {motion_aps_count + motion_evs_count}\n")
        if aps_timestamps:
            f.write(f"运动期间APS时间戳范围: {aps_timestamps[0]['timestamp_us']} - {aps_timestamps[-1]['timestamp_us']} μs\n")
        if evs_timestamps:
            f.write(f"运动期间EVS时间戳范围: {evs_timestamps[0]['timestamp_us']} - {evs_timestamps[-1]['timestamp_us']} μs\n")
        f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 写入详细的时间戳列表
        if aps_timestamps:
            f.write("\n=== 运动期间APS数据详细时间戳列表 ===\n")
            f.write("帧号\tAPS时间戳(μs)\t\t系统时间戳\t\t\t时间戳(秒)\n")
            f.write("-" * 60 + "\n")
            for aps_info in aps_timestamps:
                f.write(f"{aps_info['frame_number']}\t{aps_info['timestamp_us']}\t\t{aps_info['system_time']:.6f}\t{aps_info['timestamp_us']/1000000.0:.6f}\n")
        
          
        # 查找时间戳差距小于2毫秒的APS和EVS配对数据
        f.write("\n=== APS和EVS时间戳配对分析（差距小于2毫秒） ===\n")
        f.write("配对准则：时间戳差距 < 2000 μs (2毫秒)\n")
        f.write("-" * 80 + "\n")
        f.write("APS帧号\tAPS时间戳(μs)\t\tEVS帧号\tEVS时间戳(μs)\t\t时间差(μs)\tEVS类型\n")
        f.write("-" * 80 + "\n")
        
        paired_count = 0
        time_threshold = 2000  # 2毫秒 = 2000微秒
        
        for aps_info in aps_timestamps:
            aps_timestamp = aps_info['timestamp_us']
            
            # 寻找时间戳差距小于2毫秒的EVS数据，只保存第一个找到的配对
            best_match = None
            min_time_diff = float('inf')
            
            for evs_info in evs_timestamps:
                evs_timestamp = evs_info['timestamp_us']
                time_diff = abs(aps_timestamp - evs_timestamp)
                
                if time_diff < time_threshold and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = (evs_info, time_diff)
            
            # 如果找到配对，记录最佳配对
            if best_match:
                paired_count += 1
                evs_info, time_diff = best_match
                f.write(f"{aps_info['frame_number']}\t{aps_timestamp}\t\t{evs_info['frame_number']}\t{evs_info['timestamp_us']}\t\t{time_diff}\t\t{evs_info['data_type']}\n")
                
                # 调试信息：打印配对详情
                print(f"配对: APS帧{aps_info['frame_number']}({aps_timestamp}) <-> EVS帧{evs_info['frame_number']}({evs_timestamp}), 差值: {time_diff}μs")
                
                # 保存所有配对的APS和EVS数据为PNG文件
                try:
                    # 获取输出目录和创建配对数据文件夹
                    output_dir = os.path.dirname(output_file_path)
                    paired_data_dir = os.path.join(output_dir, "paired_data")
                    
                    # 创建配对数据文件夹（如果不存在）
                    if not os.path.exists(paired_data_dir):
                        os.makedirs(paired_data_dir)
                        print(f"创建配对数据文件夹: {paired_data_dir}")
                    
                    # 保存APS图像
                    if aps_info['frame_number'] in raw_aps_data:
                        aps_data = raw_aps_data[aps_info['frame_number']]
                        aps_image = aps_data.convertTo()
                        aps_png_path = os.path.join(paired_data_dir, f"pair_{paired_count:03d}_aps_frame{aps_info['frame_number']}.png")
                        cv.imwrite(aps_png_path, aps_image)
                        print(f"保存配对{paired_count} APS图像: {aps_png_path}")
                    
                    # 保存EVS图像
                    if evs_info['frame_number'] in raw_evs_data:
                        evs_data = raw_evs_data[evs_info['frame_number']]
                        try:
                            # 获取EVS的渲染帧（使用参考代码中的方法）
                            evs_image = evs_data.frame()
                            
                            if evs_image is not None and hasattr(evs_image, 'shape'):
                                # 根据参考代码，EVS图像需要乘以100来放大显示
                                # 但保存时需要确保数据类型正确
                                evs_display_image = evs_image * 100
                                
                                # 确保图像数据在正确范围内
                                if evs_display_image.dtype != np.uint8:
                                    # 归一化到0-255范围并转换为uint8
                                    if evs_display_image.max() > 0:
                                        evs_display_image = (evs_display_image / evs_display_image.max() * 255).astype(np.uint8)
                                    else:
                                        evs_display_image = evs_display_image.astype(np.uint8)
                                
                                # 如果是单通道图像，转换为3通道BGR格式
                                if len(evs_display_image.shape) == 2:
                                    evs_display_image = cv.cvtColor(evs_display_image, cv.COLOR_GRAY2BGR)
                                
                                # 保存EVS图像
                                evs_png_path = os.path.join(paired_data_dir, f"pair_{paired_count:03d}_evs_frame{evs_info['frame_number']}.png")
                                cv.imwrite(evs_png_path, evs_display_image)
                                print(f"保存配对{paired_count} EVS图像: {evs_png_path}")
                                
                                # 打印图像信息用于调试
                                print(f"  EVS图像信息 - 形状: {evs_display_image.shape}, 数据类型: {evs_display_image.dtype}, 值范围: {evs_display_image.min()}-{evs_display_image.max()}")
                            else:
                                print(f"警告: EVS帧{evs_info['frame_number']}的frame()方法返回无效数据")
                                
                        except Exception as e:
                            print(f"保存配对{paired_count} EVS图像失败: {e}")
                            import traceback
                            traceback.print_exc()
                    
                except Exception as e:
                    print(f"保存配对{paired_count}图像失败: {e}")
        
        f.write("-" * 80 + "\n")
        f.write(f"找到 {paired_count} 对时间戳差距小于2毫秒的APS-EVS配对数据\n")
        
        if paired_count > 0:
            # 分析配对数据的时间差分布
            f.write("\n=== 配对数据时间差分析 ===\n")
            f.write("-" * 50 + "\n")
            f.write("时间差范围(μs)\t配对数量\n")
            f.write("-" * 50 + "\n")
            
            # 统计不同时间差范围的配对数量
            diff_ranges = [
                (0, 500),      # 0-0.5ms
                (500, 1000),   # 0.5-1ms
                (1000, 1500),  # 1-1.5ms
                (1500, 2000)   # 1.5-2ms
            ]
            
            for min_diff, max_diff in diff_ranges:
                range_count = 0
                for aps_info in aps_timestamps:
                    aps_timestamp = aps_info['timestamp_us']
                    for evs_info in evs_timestamps:
                        evs_timestamp = evs_info['timestamp_us']
                        time_diff = abs(aps_timestamp - evs_timestamp)
                        if min_diff <= time_diff < max_diff:
                            range_count += 1
                
                f.write(f"{min_diff}-{max_diff}\t\t{range_count}\n")
            
            # 计算平均时间差
            total_diff = 0
            pair_count = 0
            for aps_info in aps_timestamps:
                aps_timestamp = aps_info['timestamp_us']
                for evs_info in evs_timestamps:
                    evs_timestamp = evs_info['timestamp_us']
                    time_diff = abs(aps_timestamp - evs_timestamp)
                    if time_diff < time_threshold:
                        total_diff += time_diff
                        pair_count += 1
            
            if pair_count > 0:
                avg_diff = total_diff / pair_count
                f.write(f"\n平均时间差: {avg_diff:.2f} μs ({avg_diff/1000:.3f} ms)\n")
                f.write(f"最小时间差: {min(abs(aps_info['timestamp_us'] - evs_info['timestamp_us']) for aps_info in aps_timestamps for evs_info in evs_timestamps if abs(aps_info['timestamp_us'] - evs_info['timestamp_us']) < time_threshold):.2f} μs\n")
                f.write(f"最大时间差: {max(abs(aps_info['timestamp_us'] - evs_info['timestamp_us']) for aps_info in aps_timestamps for evs_info in evs_timestamps if abs(aps_info['timestamp_us'] - evs_info['timestamp_us']) < time_threshold):.2f} μs\n")
    
    print(f"运动期间数据已保存到: {output_file_path}")
    print(f"总共处理了 {frame_count} 帧数据")
    print(f"运动期间APS数据数量: {motion_aps_count}")
    print(f"运动期间EVS数据数量: {motion_evs_count}")
    print(f"运动期间数据总数: {motion_aps_count + motion_evs_count}")
    
    if aps_timestamps:
        print(f"运动期间APS时间戳范围: {aps_timestamps[0]['timestamp_us']} - {aps_timestamps[-1]['timestamp_us']} μs")
        print(f"对应的系统时间范围: {aps_timestamps[0]['system_time']:.6f} - {aps_timestamps[-1]['system_time']:.6f}")
    
    if evs_timestamps:
        print(f"运动期间EVS时间戳范围: {evs_timestamps[0]['timestamp_us']} - {evs_timestamps[-1]['timestamp_us']} μs")
        print(f"对应的系统时间范围: {evs_timestamps[0]['system_time']:.6f} - {evs_timestamps[-1]['system_time']:.6f}")
        
        # 调试信息：打印前5个EVS时间戳
        print(f"前5个EVS时间戳:")
        for i, evs_info in enumerate(evs_timestamps[:5]):
            print(f"  EVS帧{evs_info['frame_number']}: {evs_info['timestamp_us']} μs")
    
    if not aps_timestamps and not evs_timestamps:
        print("未找到运动期间的数据")
        print("可能的原因：")
        print("1. 时间戳对齐不准确")
        print("2. 运动期间没有采集到数据")
        print("3. 数据采集时间与运动时间不匹配")

def extractDataInfo(player, output_file_path, data_file_path):
    """
    提取数据的时间戳、数据类型和数据尺寸信息并保存到txt文件
    
    :param player: 播放器实例
    :param output_file_path: 输出txt文件路径
    :param data_file_path: 数据文件路径
    """
    frame_count = 0
    total_frames = 0
    
    # 首先获取总帧数
    try:
        total_frames = player.getTotalFrames()
        print(f"Total frames in data: {total_frames}")
    except:
        print("Could not get total frame count")
        total_frames = 0
    
    # 创建输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("=== AlpLib数据信息提取结果 ===\n")
        f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件路径: {data_file_path}\n")
        f.write(f"总帧数: {total_frames}\n")
        f.write("-" * 50 + "\n")
        f.write("帧号\t时间戳\t\t\t数据类型\t数据尺寸/事件信息\n")
        f.write("-" * 50 + "\n")
        
        # 持续监控设备状态并提取数据信息
        while player.isWorking():
            # 获取同步数据（包含APS和EVS）
            sync_list = player.getSyncFrames()
            
            for it in sync_list:
                # 获取APS数据信息
                if len(it) > 0 and it[0] is not None:
                    frame_count += 1
                    aps_info = extract_aps_info(it[0], frame_count)
                    
                    if aps_info:
                        timestamp_str = f"{aps_info['timestamp_us']} μs"
                        f.write(f"{frame_count}\t{timestamp_str}\t{aps_info['data_type']}\t{aps_info['size_str']}\n")
                        f.flush()
                    else:
                        f.write(f"{frame_count}\tN/A\tAPS\tN/A\n")
                        f.flush()
                    
                    # 打印进度
                    if frame_count % 100 == 0:
                        print(f"已处理 {frame_count} 帧...")
                
                # 获取EVS数据信息
                if len(it) > 1 and len(it[1]) > 0:
                    for evs_data in it[1]:
                        frame_count += 1
                        evs_info = extract_evs_info(evs_data, frame_count)
                        
                        if evs_info:
                            timestamp_str = f"{evs_info['timestamp_us']} μs"
                            if evs_info['data_type'] == 'EVS_Events':
                                f.write(f"{frame_count}\t{timestamp_str}\t{evs_info['data_type']}\t{evs_info['size_str']}\n")
                            else:
                                f.write(f"{frame_count}\t{timestamp_str}\t{evs_info['data_type']}\t{evs_info['size_str']}\n")
                            f.flush()
                        else:
                            f.write(f"{frame_count}\tN/A\tEVS\tN/A\n")
                            f.flush()
            
            # 避免过度占用CPU
            time.sleep(0.001)
        
        f.write("-" * 50 + "\n")
        f.write(f"处理完成，总共处理了 {frame_count} 帧数据\n")
        f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"数据信息已保存到: {output_file_path}")
    print(f"总共处理了 {frame_count} 帧数据")

def loadData(player, path, model=3):
    """
    加载保存的 alpdata 数据
    
    :param player: 播放器实例
    :param path: 播放文件路径
    :param model: 数据模式 (1=APS, 2=EVS, 3=HVS/同步)
    """
    
    if ".alpdata" in path:
        ## 初始化播放器 ##
        return player.init(PlayAlpDataType(model), path), model
    else:
        return False, model

def find_data_files(folder_path):
    """
    在指定文件夹中查找alpdata文件和sync_timestamps.txt文件
    
    :param folder_path: 文件夹路径
    :return: (data_file_path, sync_timestamps_path) 或 (None, None)如果未找到
    """
    try:
        # 查找所有的alpdata文件
        alpdata_files = []
        sync_files = []
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if file.endswith('.alpdata'):
                    alpdata_files.append(file_path)
                elif file == 'sync_timestamps.txt':
                    sync_files.append(file_path)
        
        # 如果找到多个alpdata文件，选择最新的一个
        if alpdata_files:
            # 按修改时间排序，选择最新的文件
            alpdata_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            data_file_path = alpdata_files[0]
            
            if sync_files:
                sync_timestamps_path = sync_files[0]
                return data_file_path, sync_timestamps_path
            else:
                print(f"警告: 在文件夹 {folder_path} 中找到alpdata文件但未找到sync_timestamps.txt")
                return data_file_path, None
        else:
            print(f"在文件夹 {folder_path} 中未找到.alpdata文件")
            return None, None
            
    except Exception as e:
        print(f"扫描文件夹时出错: {e}")
        return None, None

def main():
    """
    主函数：读取指定的alpdata文件并提取机械臂运动期间的APS和EVS数据
    """
    
    # 获取用户输入的文件夹路径
    print("请输入数据文件夹路径（包含.alpdata文件和sync_timestamps.txt）:")
    folder_path = input().strip()
    
    # 去除路径两端的引号（如果有的话）
    if folder_path.startswith('"') and folder_path.endswith('"'):
        folder_path = folder_path[1:-1]
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        return
    
    if not os.path.isdir(folder_path):
        print(f"错误: {folder_path} 不是一个文件夹")
        return
    
    # 自动查找数据文件
    data_file_path, sync_timestamps_path = find_data_files(folder_path)
    
    if data_file_path is None:
        print("未找到数据文件，程序退出")
        return
    
    if sync_timestamps_path is None:
        print("未找到sync_timestamps.txt文件，程序退出")
        return
    
    ## 一. 播放器实例化 ##
    player = AlpPlayer()
    
    # 生成输出文件路径
    output_dir = os.path.dirname(data_file_path)
    input_filename = os.path.splitext(os.path.basename(data_file_path))[0]
    output_txt_path = os.path.join(output_dir, f"{input_filename}_motion_aps_data.txt")
    
    print(f"找到数据文件: {data_file_path}")
    print(f"找到同步文件: {sync_timestamps_path}")
    print(f"输出文件路径: {output_txt_path}")
    
    ## 二. 初始化播放器 ##
    success, model = loadData(player, data_file_path, 3)  # 使用同步模式（HVS）
    
    if not success:
        print("加载数据失败，退出...")
        return
    
    print("数据加载成功，开始提取运动期间数据...")
    
    ## 三. 创建并启动信息提取线程 ##
    extract_thread = threading.Thread(target=extractMotionData, args=(player, output_txt_path, data_file_path, sync_timestamps_path))
    extract_thread.start()
    
    ## 四. 加载数据 ##
    if not player.load():
        print("加载播放器失败，退出...")
        return
    
    ## 五. 播放数据流 ##
    if not player.play():
        print("开始播放失败，退出...")
        return
    
    ## 六. 监控设备状态 ##
    print("正在提取运动期间数据...")
    while player.isWorking():
        time.sleep(0.1)
    
    ## 七. 关闭播放器 ##
    player.close()
    
    # 等待提取线程完成
    extract_thread.join()
    
    print("运动期间数据提取完成！")

if __name__ == "__main__":
    main()