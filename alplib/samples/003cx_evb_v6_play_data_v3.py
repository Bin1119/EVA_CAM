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

# 全局配置变量
MAX_PAIRED_DATA_COUNT = 10  # 配对数据保留的最大数量

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
        
        # 收集所有配对数据用于创建筛选后的alpdata文件
        paired_aps_evs_data = []
        
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
                
                # 收集配对数据信息
                paired_aps_evs_data.append({
                    'aps_frame': aps_info['frame_number'],
                    'evs_frame': evs_info['frame_number'],
                    'time_diff': time_diff,
                    'aps_timestamp': aps_timestamp,
                    'evs_timestamp': evs_info['timestamp_us']
                })
                
                # 调试信息：打印配对详情
                print(f"配对: APS帧{aps_info['frame_number']}({aps_timestamp}) <-> EVS帧{evs_info['frame_number']}({evs_info['timestamp_us']}), 差值: {time_diff}μs")
                
                # 保存所有配对的APS和EVS数据为PNG文件
                try:
                    # 获取输出目录和创建配对数据文件夹
                    output_dir = os.path.dirname(output_file_path)
                    paired_data_dir = os.path.join(output_dir, "paired_data")
                    
                    # 创建配对数据文件夹（如果不存在）
                    if not os.path.exists(paired_data_dir):
                        os.makedirs(paired_data_dir)
                        print(f"创建配对数据文件夹: {paired_data_dir}")
                    
                    # 拼接并保存APS和EVS配对图像
                    if aps_info['frame_number'] in raw_aps_data and evs_info['frame_number'] in raw_evs_data:
                        aps_data = raw_aps_data[aps_info['frame_number']]
                        evs_data = raw_evs_data[evs_info['frame_number']]
                        
                        try:
                            # 获取APS图像
                            aps_image = aps_data.convertTo()
                            
                            # 获取EVS图像
                            evs_image = evs_data.frame()
                            
                            if evs_image is not None and hasattr(evs_image, 'shape'):
                                # 处理EVS图像
                                evs_display_image = evs_image * 100
                                
                                # 确保图像数据在正确范围内
                                if evs_display_image.dtype != np.uint8:
                                    if evs_display_image.max() > 0:
                                        evs_display_image = (evs_display_image / evs_display_image.max() * 255).astype(np.uint8)
                                    else:
                                        evs_display_image = evs_display_image.astype(np.uint8)
                                
                                # 如果是单通道图像，转换为3通道BGR格式
                                if len(evs_display_image.shape) == 2:
                                    evs_display_image = cv.cvtColor(evs_display_image, cv.COLOR_GRAY2BGR)
                                
                                # 确保APS图像也是3通道BGR格式
                                if len(aps_image.shape) == 2:
                                    aps_image = cv.cvtColor(aps_image, cv.COLOR_GRAY2BGR)
                                
                                # 调整图像大小以便拼接（统一到较小的尺寸）
                                h1, w1 = aps_image.shape[:2]
                                h2, w2 = evs_display_image.shape[:2]
                                
                                # 计算统一高度（取较小值）
                                target_height = min(h1, h2)
                                
                                # 调整APS图像大小
                                aps_resized = cv.resize(aps_image, (int(w1 * target_height / h1), target_height))
                                
                                # 调整EVS图像大小
                                evs_resized = cv.resize(evs_display_image, (int(w2 * target_height / h2), target_height))
                                
                                # 水平拼接图像
                                paired_image = cv.hconcat([aps_resized, evs_resized])
                                
                                # 添加标签
                                cv.putText(paired_image, f"APS Frame {aps_info['frame_number']}", (10, 30), 
                                         cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv.putText(paired_image, f"EVS Frame {evs_info['frame_number']}", (aps_resized.shape[1] + 10, 30), 
                                         cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv.putText(paired_image, f"Time Diff: {time_diff}us", (aps_resized.shape[1] + 10, 60), 
                                         cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # 保存拼接后的图像
                                paired_png_path = os.path.join(paired_data_dir, f"pair_{paired_count:03d}_aps{aps_info['frame_number']}_evs{evs_info['frame_number']}.png")
                                cv.imwrite(paired_png_path, paired_image)
                                print(f"保存配对{paired_count} 拼接图像: {paired_png_path}")
                                
                                # 清理多余的配对数据文件，只保留最后MAX_PAIRED_DATA_COUNT个
                                cleanup_excess_paired_data(paired_data_dir, paired_count)
                                
                                # 打印图像信息用于调试
                                print(f"  APS图像信息 - 形状: {aps_image.shape}, 数据类型: {aps_image.dtype}")
                                print(f"  EVS图像信息 - 形状: {evs_display_image.shape}, 数据类型: {evs_display_image.dtype}, 值范围: {evs_display_image.min()}-{evs_display_image.max()}")
                                print(f"  拼接图像信息 - 形状: {paired_image.shape}")
                            else:
                                print(f"警告: EVS帧{evs_info['frame_number']}的frame()方法返回无效数据")
                                
                        except Exception as e:
                            print(f"保存配对{paired_count} 拼接图像失败: {e}")
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
    
    # 新增功能：保存配对数据的APS RAW和EVS txyp数据
    if paired_count > 0 and paired_aps_evs_data and raw_aps_data and raw_evs_data:
        print("\n=== 开始保存配对RAW和EVS数据 ===")
        # 获取输出目录
        output_dir = os.path.dirname(output_file_path)
        
        # 保存配对数据的APS RAW和EVS txyp数据
        save_paired_raw_and_evs_data(raw_aps_data, raw_evs_data, aps_timestamps, output_dir, MAX_PAIRED_DATA_COUNT)

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

def cleanup_excess_paired_data(paired_data_dir, current_paired_count):
    """
    清理多余的配对数据文件，只保留最后MAX_PAIRED_DATA_COUNT个
    
    :param paired_data_dir: 配对数据目录
    :param current_paired_count: 当前配对数量
    """
    global MAX_PAIRED_DATA_COUNT
    
    # 如果当前配对数量小于等于最大保留数量，不需要清理
    if current_paired_count <= MAX_PAIRED_DATA_COUNT:
        return
    
    # 获取所有配对数据文件
    paired_files = []
    if os.path.exists(paired_data_dir):
        for filename in os.listdir(paired_data_dir):
            if filename.startswith('pair_') and filename.endswith('.png'):
                file_path = os.path.join(paired_data_dir, filename)
                paired_files.append(file_path)
    
    # 按文件名排序（这样就是按时间顺序）
    paired_files.sort()
    
    # 计算需要删除的文件数量
    files_to_delete = len(paired_files) - MAX_PAIRED_DATA_COUNT
    
    if files_to_delete > 0:
        print(f"清理配对数据：删除 {files_to_delete} 个旧文件，保留最新的 {MAX_PAIRED_DATA_COUNT} 个")
        
        # 删除最旧的文件
        for i in range(files_to_delete):
            try:
                os.remove(paired_files[i])
                print(f"  已删除: {os.path.basename(paired_files[i])}")
            except Exception as e:
                print(f"  删除失败 {paired_files[i]}: {e}")

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

def find_paired_collection_files(session_folder_path):
    """
    在配对采集会话文件夹中查找两个数据采集文件和相关信息
    
    :param session_folder_path: 配对采集会话文件夹路径
    :return: 包含两个采集文件信息的字典或None
    """
    try:
        # 检查会话文件夹结构
        collection_files = []
        session_info = None
        collection_01_info = None
        collection_02_info = None
        sync_timestamps = None
        
        for file in os.listdir(session_folder_path):
            file_path = os.path.join(session_folder_path, file)
            if os.path.isfile(file_path):
                if file.endswith('.alpdata'):
                    collection_files.append(file_path)
                elif file == 'session_info.txt':
                    session_info = file_path
                elif file == 'collection_01_info.txt':
                    collection_01_info = file_path
                elif file == 'collection_02_info.txt':
                    collection_02_info = file_path
                elif file == 'sync_timestamps.txt':
                    sync_timestamps = file_path
        
        # 验证文件完整性
        if len(collection_files) != 2:
            print(f"错误: 在配对采集会话文件夹中找到 {len(collection_files)} 个数据文件，期望2个")
            return None
            
        if not all([session_info, collection_01_info, collection_02_info, sync_timestamps]):
            print(f"错误: 缺少必要的配置文件")
            return None
        
        # 按文件名排序确定采集顺序
        collection_files.sort()
        
        return {
            'collection_01': collection_files[0],
            'collection_02': collection_files[1],
            'session_info': session_info,
            'collection_01_info': collection_01_info,
            'collection_02_info': collection_02_info,
            'sync_timestamps': sync_timestamps,
            'session_folder': session_folder_path
        }
        
    except Exception as e:
        print(f"扫描配对采集会话文件夹时出错: {e}")
        return None

def find_data_files(folder_path):
    """
    在指定文件夹中查找数据文件（支持单文件和配对采集会话模式）
    
    :param folder_path: 文件夹路径
    :return: 根据文件类型返回相应的数据结构
    """
    try:
        # 检查是否为配对采集会话文件夹
        if 'paired_collection_session_' in os.path.basename(folder_path):
            print(f"检测到配对采集会话文件夹: {folder_path}")
            paired_data = find_paired_collection_files(folder_path)
            if paired_data:
                return paired_data
            else:
                print("配对采集会话文件不完整")
                return None
        
        # 原有的单文件查找逻辑
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
            alpdata_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            data_file_path = alpdata_files[0]
            
            if sync_files:
                sync_timestamps_path = sync_files[0]
                return {
                    'mode': 'single',
                    'data_file': data_file_path,
                    'sync_timestamps': sync_timestamps_path
                }
            else:
                print(f"警告: 在文件夹 {folder_path} 中找到alpdata文件但未找到sync_timestamps.txt")
                return {
                    'mode': 'single',
                    'data_file': data_file_path,
                    'sync_timestamps': None
                }
        else:
            print(f"在文件夹 {folder_path} 中未找到.alpdata文件")
            return None
            
    except Exception as e:
        print(f"扫描文件夹时出错: {e}")
        return None

def extract_paired_collection_data(paired_data_info):
    """
    提取配对采集会话中的两个采集数据并进行对比分析
    
    :param paired_data_info: 配对采集数据信息字典
    """
    print("开始处理配对采集数据...")
    
    # 读取会话信息
    session_data = read_session_info(paired_data_info['session_info'])
    collection_01_data = read_collection_info(paired_data_info['collection_01_info'])
    collection_02_data = read_collection_info(paired_data_info['collection_02_info'])
    
    if not all([session_data, collection_01_data, collection_02_data]):
        print("读取会话信息失败")
        return
    
    # 显示配对采集信息
    print("\n=== 配对采集会话信息 ===")
    print(f"会话时间: {session_data.get('会话开始时间', 'N/A')}")
    print(f"初始位置: {session_data.get('初始位置', 'N/A')}")
    print(f"总采集次数: {session_data.get('总采集次数', 'N/A')}")
    
    print("\n=== 第一次采集信息 ===")
    print(f"采集时间: {collection_01_data.get('采集时间', 'N/A')}")
    print(f"运动持续时间: {collection_01_data.get('运动持续时间', 'N/A')}秒")
    print(f"起始位置: {collection_01_data.get('起始位置', 'N/A')}")
    print(f"结束位置: {collection_01_data.get('结束位置', 'N/A')}")
    print(f"总移动距离: {collection_01_data.get('总移动距离', 'N/A')}")
    print(f"运动方向: {collection_01_data.get('运动方向（YZ平面）', 'N/A')}")
    
    print("\n=== 第二次采集信息 ===")
    print(f"采集时间: {collection_02_data.get('采集时间', 'N/A')}")
    print(f"运动持续时间: {collection_02_data.get('运动持续时间', 'N/A')}秒")
    print(f"起始位置: {collection_02_data.get('起始位置', 'N/A')}")
    print(f"结束位置: {collection_02_data.get('结束位置', 'N/A')}")
    print(f"总移动距离: {collection_02_data.get('总移动距离', 'N/A')}")
    print(f"运动方向: {collection_02_data.get('运动方向（YZ平面）', 'N/A')}")
    
    # 对比分析
    print("\n=== 配对采集对比分析 ===")
    duration_01 = float(collection_01_data.get('运动持续时间', '0').replace('秒', '').strip())
    duration_02 = float(collection_02_data.get('运动持续时间', '0').replace('秒', '').strip())
    distance_01 = float(collection_01_data.get('总移动距离', '0').replace('mm', '').strip())
    distance_02 = float(collection_02_data.get('总移动距离', '0').replace('mm', '').strip())
    
    duration_diff = abs(duration_01 - duration_02)
    distance_diff = abs(distance_01 - distance_02)
    
    print(f"运动时间差: {duration_diff:.6f}秒")
    print(f"运动距离差: {distance_diff:.3f}mm")
    print(f"时间一致性: {'优秀' if duration_diff < 0.01 else '良好' if duration_diff < 0.05 else '需改进'}")
    print(f"距离一致性: {'优秀' if distance_diff < 0.1 else '良好' if distance_diff < 1.0 else '需改进'}")
    
    # 分别处理两个采集文件
    output_dir = paired_data_info['session_folder']
    
    # 为每次采集创建单独的分析文件夹
    collection_01_analysis_dir = os.path.join(output_dir, "collection_01_analysis")
    collection_02_analysis_dir = os.path.join(output_dir, "collection_02_analysis")
    
    os.makedirs(collection_01_analysis_dir, exist_ok=True)
    os.makedirs(collection_02_analysis_dir, exist_ok=True)
    
    # 处理第一次采集
    print("\n=== 处理第一次采集数据 ===")
    output_01_path = os.path.join(collection_01_analysis_dir, "analysis.txt")
    extractMotionDataForCollection(paired_data_info['collection_01'], 
                                   output_01_path, 
                                   paired_data_info['sync_timestamps'],
                                   "第一次采集")
    
    # 处理第二次采集
    print("\n=== 处理第二次采集数据 ===")
    output_02_path = os.path.join(collection_02_analysis_dir, "analysis.txt")
    extractMotionDataForCollection(paired_data_info['collection_02'], 
                                   output_02_path, 
                                   paired_data_info['sync_timestamps'],
                                   "第二次采集")
    
    # 生成对比报告
    generate_comparison_report(output_dir, collection_01_data, collection_02_data)
    
    print("\n配对采集数据分析完成！")

def read_session_info(session_info_path):
    """
    读取会话信息文件
    
    :param session_info_path: 会话信息文件路径
    :return: 会话信息字典
    """
    try:
        info = {}
        with open(session_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('==='):
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
        return info
    except Exception as e:
        print(f"读取会话信息失败: {e}")
        return None

def read_collection_info(collection_info_path):
    """
    读取采集信息文件
    
    :param collection_info_path: 采集信息文件路径
    :return: 采集信息字典
    """
    try:
        info = {}
        with open(collection_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('==='):
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
        return info
    except Exception as e:
        print(f"读取采集信息失败: {e}")
        return None

def extractMotionDataForCollection(data_file_path, output_file_path, sync_timestamps_path, collection_name):
    """
    为单个采集文件提取运动期间的数据
    
    :param data_file_path: 数据文件路径
    :param output_file_path: 输出文件路径
    :param sync_timestamps_path: 同步时间戳文件路径
    :param collection_name: 采集名称（用于显示）
    """
    ## 一. 播放器实例化 ##
    player = AlpPlayer()
    
    ## 二. 初始化播放器 ##
    success, model = loadData(player, data_file_path, 3)  # 使用同步模式（HVS）
    
    if not success:
        print(f"加载数据失败 ({collection_name})，退出...")
        return
    
    print(f"数据加载成功 ({collection_name})，开始提取运动期间数据...")
    
    ## 三. 创建并启动信息提取线程 ##
    extract_thread = threading.Thread(target=extractMotionData, args=(player, output_file_path, data_file_path, sync_timestamps_path))
    extract_thread.start()
    
    ## 四. 加载数据 ##
    if not player.load():
        print(f"加载播放器失败 ({collection_name})，退出...")
        return
    
    ## 五. 播放数据流 ##
    if not player.play():
        print(f"开始播放失败 ({collection_name})，退出...")
        return
    
    ## 六. 监控设备状态 ##
    print(f"正在提取运动期间数据 ({collection_name})...")
    while player.isWorking():
        time.sleep(0.1)
    
    ## 七. 关闭播放器 ##
    player.close()
    
    # 等待提取线程完成
    extract_thread.join()
    
    print(f"运动期间数据提取完成 ({collection_name})！")

def generate_comparison_report(output_dir, collection_01_data, collection_02_data):
    """
    生成配对采集对比报告
    
    :param output_dir: 输出目录
    :param collection_01_data: 第一次采集数据
    :param collection_02_data: 第二次采集数据
    """
    report_path = os.path.join(output_dir, "paired_collection_comparison_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 配对采集对比分析报告 ===\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("=== 第一次采集信息 ===\n")
        for key, value in collection_01_data.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== 第二次采集信息 ===\n")
        for key, value in collection_02_data.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== 一致性分析 ===\n")
        duration_01 = float(collection_01_data.get('运动持续时间', '0').replace('秒', '').strip())
        duration_02 = float(collection_02_data.get('运动持续时间', '0').replace('秒', '').strip())
        distance_01 = float(collection_01_data.get('总移动距离', '0').replace('mm', '').strip())
        distance_02 = float(collection_02_data.get('总移动距离', '0').replace('mm', '').strip())
        
        duration_diff = abs(duration_01 - duration_02)
        distance_diff = abs(distance_01 - distance_02)
        duration_percent = (duration_diff / duration_01 * 100) if duration_01 > 0 else 0
        distance_percent = (distance_diff / distance_01 * 100) if distance_01 > 0 else 0
        
        f.write(f"运动时间差: {duration_diff:.6f}秒 ({duration_percent:.2f}%)\n")
        f.write(f"运动距离差: {distance_diff:.3f}mm ({distance_percent:.2f}%)\n")
        f.write(f"时间一致性评级: {'优秀' if duration_diff < 0.01 else '良好' if duration_diff < 0.05 else '需改进'}\n")
        f.write(f"距离一致性评级: {'优秀' if distance_diff < 0.1 else '良好' if distance_diff < 1.0 else '需改进'}\n")
        f.write("\n")
        
        f.write("=== 评估结果 ===\n")
        if duration_diff < 0.01 and distance_diff < 0.1:
            f.write("评估结果: 优秀 - 两次采集具有高度一致性\n")
        elif duration_diff < 0.05 and distance_diff < 1.0:
            f.write("评估结果: 良好 - 两次采集基本一致\n")
        else:
            f.write("评估结果: 需改进 - 两次采集存在明显差异\n")
        
    print(f"对比报告已保存到: {report_path}")

def save_aps_as_10bit_raw(aps_data, output_path):
    """
    将APS数据保存为10bit RAW格式
    
    :param aps_data: APS数据对象
    :param output_path: 输出文件路径
    """
    try:
        # 获取APS图像
        aps_image = aps_data.convertTo()
        
        # 确保数据是16位（可以容纳10bit数据）
        if aps_image.dtype != np.uint16:
            if aps_image.dtype == np.uint8:
                # 8位转10位（左移2位）
                aps_image = (aps_image.astype(np.uint16) << 2)
            else:
                # 其他类型直接转换为16位
                aps_image = aps_image.astype(np.uint16)
        
        # 保存为RAW文件（二进制格式）
        with open(output_path, 'wb') as f:
            aps_image.tofile(f)
        
        print(f"APS 10bit RAW数据已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"保存APS 10bit RAW数据失败: {e}")
        return False

def extract_evs_txyp_from_frame(evs_data):
    """
    从EVS帧中提取txyp格式数据
    
    :param evs_data: EVS数据对象
    :return: txyp数据列表 [(t, x, y, p), ...]
    """
    try:
        # 获取帧的时间戳
        timestamp = evs_data.timestamp()
        
        # 获取事件点数据
        event_points = evs_data.point()
        
        # 提取txyp数据
        txyp_data = []
        for event in event_points:
            txyp_data.append([timestamp, event.x, event.y, ord(event.p)])
        
        return txyp_data
        
    except Exception as e:
        print(f"提取EVS txyp数据失败: {e}")
        return []

def save_evs_txyp_intervals(evs_frames_dict, output_dir, aps_timestamps, max_count=10):
    """
    保存配对数据中指定数量APS时间戳之间的EVS数据为多个txyp格式的npy文件
    
    :param evs_frames_dict: EVS帧字典 {帧号: EVS数据}
    :param output_dir: 输出目录
    :param aps_timestamps: APS时间戳列表 [{'frame_number': ..., 'timestamp_us': ...}, ...]
    :param max_count: 最大保存的APS帧数量
    """
    try:
        # 创建EVS数据输出目录
        evs_output_dir = os.path.join(output_dir, "evs_txyp_data")
        os.makedirs(evs_output_dir, exist_ok=True)
        
        # 如果APS时间戳少于2个，无法分割
        if len(aps_timestamps) < 2:
            print("APS时间戳不足，无法分割EVS数据")
            return False
        
        # 只保留最后max_count个APS时间戳
        if len(aps_timestamps) > max_count:
            aps_timestamps = aps_timestamps[-max_count:]
            print(f"只保留最后{max_count}个APS时间戳的EVS数据")
        
        # 按时间戳排序EVS数据
        sorted_evs_frames = []
        for frame_number, evs_data in evs_frames_dict.items():
            txyp_data = extract_evs_txyp_from_frame(evs_data)
            if txyp_data:
                sorted_evs_frames.extend(txyp_data)
        
        # 按时间戳排序EVS事件
        sorted_evs_frames.sort(key=lambda x: x[0])
        
        # 获取时间范围（只处理配对数据的时间范围）
        start_timestamp = aps_timestamps[0]['timestamp_us']
        end_timestamp = aps_timestamps[-1]['timestamp_us']
        
        # 筛选时间范围内的EVS事件
        paired_range_events = []
        for event in sorted_evs_frames:
            if start_timestamp <= event[0] <= end_timestamp:
                paired_range_events.append(event)
            elif event[0] > end_timestamp:
                break
        
        print(f"配对数据时间范围: {start_timestamp}-{end_timestamp} μs")
        print(f"该时间范围内EVS事件数: {len(paired_range_events)}")
        
        # 分割EVS数据并保存，记录时间戳段信息
        saved_evs_intervals = []  # 保存每个EVS文件的时间戳段信息
        
        for i in range(len(aps_timestamps) - 1):
            # 获取当前APS帧和下一APS帧的时间戳
            current_aps = aps_timestamps[i]
            next_aps = aps_timestamps[i + 1]
            
            interval_start = current_aps['timestamp_us']
            interval_end = next_aps['timestamp_us']
            
            # 筛选时间范围内的EVS事件
            interval_events = []
            for event in paired_range_events:
                if interval_start <= event[0] <= interval_end:
                    interval_events.append(event)
                elif event[0] > interval_end:
                    break
            
            # 保存为npy文件
            if interval_events:
                interval_array = np.array(interval_events, dtype=np.int64)
                output_path = os.path.join(evs_output_dir, f"evs_{current_aps['frame_number']:03d}_{next_aps['frame_number']:03d}.npy")
                np.save(output_path, interval_array)
                
                # 记录时间戳段信息
                saved_evs_intervals.append({
                    'filename': f"evs_{current_aps['frame_number']:03d}_{next_aps['frame_number']:03d}.npy",
                    'start_frame': current_aps['frame_number'],
                    'end_frame': next_aps['frame_number'],
                    'start_timestamp_us': interval_start,
                    'end_timestamp_us': interval_end,
                    'start_timestamp_sec': interval_start / 1000000.0,
                    'end_timestamp_sec': interval_end / 1000000.0,
                    'event_count': len(interval_events),
                    'duration_sec': (interval_end - interval_start) / 1000000.0
                })
                
                print(f"保存EVS间隔数据: {current_aps['frame_number']}-{next_aps['frame_number']}, 事件数: {len(interval_events)}, 时间范围: {interval_start}-{interval_end} μs")
            else:
                print(f"警告: APS帧{current_aps['frame_number']}-{next_aps['frame_number']}之间没有EVS事件")
        
        # 保存EVS数据元数据
        metadata_path = os.path.join(evs_output_dir, "metadata.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=== EVS txyp数据提取结果 ===\n")
            f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总APS帧数: {len(aps_timestamps)}\n")
            f.write(f"保存的APS帧数: {len(aps_timestamps)}\n")
            f.write(f"EVS间隔文件数: {len(aps_timestamps) - 1}\n")
            f.write(f"配对数据时间范围: {start_timestamp}-{end_timestamp} μs\n")
            f.write(f"时间范围内EVS事件数: {len(paired_range_events)}\n")
            
            # 添加EVS分辨率和时间戳信息
            if evs_frames_dict:
                first_frame_number = sorted(evs_frames_dict.keys())[0]
                first_evs_data = evs_frames_dict[first_frame_number]
                
                # 获取EVS分辨率信息
                try:
                    # 尝试多种方法获取EVS传感器尺寸
                    evs_width = None
                    evs_height = None
                    
                    if hasattr(first_evs_data, 'getWidth') and hasattr(first_evs_data, 'getHeight'):
                        try:
                            evs_width = first_evs_data.getWidth()
                            evs_height = first_evs_data.getHeight()
                            if callable(evs_width):
                                evs_width = evs_width()
                            if callable(evs_height):
                                evs_height = evs_height()
                        except:
                            pass
                    
                    if evs_width is None or evs_height is None:
                        if hasattr(first_evs_data, 'width') and hasattr(first_evs_data, 'height'):
                            try:
                                evs_width = first_evs_data.width
                                evs_height = first_evs_data.height
                                if callable(evs_width):
                                    evs_width = evs_width()
                                if callable(evs_height):
                                    evs_height = evs_height()
                            except:
                                pass
                    
                    if evs_width is None or evs_height is None:
                        # 尝试从渲染帧获取尺寸
                        try:
                            evs_image = first_evs_data.frame()
                            if hasattr(evs_image, 'shape'):
                                height, width = evs_image.shape[:2]
                                evs_width = width
                                evs_height = height
                        except:
                            pass
                    
                    if evs_width is not None and evs_height is not None:
                        f.write(f"EVS传感器分辨率: {evs_width}x{evs_height}\n")
                    else:
                        f.write("EVS传感器分辨率: N/A\n")
                        
                except Exception as e:
                    f.write("EVS传感器分辨率: N/A\n")
                
                # 获取EVS时间戳信息
                try:
                    timestamp_us = extract_timestamp_from_data(first_evs_data)
                    if timestamp_us is not None:
                        f.write(f"第一帧时间戳: {timestamp_us} μs\n")
                        f.write(f"第一帧时间戳(秒): {timestamp_us/1000000:.6f} 秒\n")
                    else:
                        f.write("第一帧时间戳: N/A\n")
                except Exception as e:
                    f.write("第一帧时间戳: N/A\n")
                
                # 获取EVS时间戳范围
                try:
                    timestamps = []
                    for frame_number, evs_data in evs_frames_dict.items():
                        ts = extract_timestamp_from_data(evs_data)
                        if ts is not None:
                            timestamps.append(ts)
                    
                    if timestamps:
                        min_ts = min(timestamps)
                        max_ts = max(timestamps)
                        f.write(f"EVS时间戳范围: {min_ts}-{max_ts} μs\n")
                        f.write(f"EVS时间戳范围(秒): {min_ts/1000000:.6f}-{max_ts/1000000:.6f} 秒\n")
                        f.write(f"EVS采集持续时间: {(max_ts-min_ts)/1000000:.6f} 秒\n")
                        
                        # 计算EVS事件频率
                        if len(paired_range_events) > 0 and (max_ts - min_ts) > 0:
                            event_rate = len(paired_range_events) / ((max_ts - min_ts) / 1000000.0)
                            f.write(f"EVS事件频率: {event_rate:.2f} 事件/秒\n")
                    else:
                        f.write("EVS时间戳范围: N/A\n")
                except Exception as e:
                    f.write("EVS时间戳范围: N/A\n")
            
            f.write("数据格式: NumPy数组 (.npy)\n")
            f.write("数据结构: [时间戳(μs), x坐标, y坐标, 极性(1=正事件,0=负事件)]\n")
            f.write("文件命名规则: evs_XXX_YYY.npy (XXX=起始APS帧号, YYY=结束APS帧号)\n")
            
            # 添加所有保存的EVS文件时间戳段详细信息
            if saved_evs_intervals:
                f.write("\n=== 保存的EVS文件详细时间戳段信息 ===\n")
                f.write("起始帧\t结束帧\t文件名\t\t\t\t时间戳范围(μs)\t\t时间戳范围(秒)\t\t事件数\t持续时间(秒)\n")
                f.write("-" * 110 + "\n")
                
                for evs_interval in saved_evs_intervals:
                    start_frame = evs_interval['start_frame']
                    end_frame = evs_interval['end_frame']
                    filename = evs_interval['filename']
                    start_ts = evs_interval['start_timestamp_us']
                    end_ts = evs_interval['end_timestamp_us']
                    start_sec = evs_interval['start_timestamp_sec']
                    end_sec = evs_interval['end_timestamp_sec']
                    event_count = evs_interval['event_count']
                    duration = evs_interval['duration_sec']
                    
                    f.write(f"{start_frame:03d}\t{end_frame:03d}\t{filename}\t\t{start_ts}-{end_ts}\t\t{start_sec:.6f}-{end_sec:.6f}\t{event_count}\t{duration:.6f}\n")
        
        print(f"EVS txyp数据已保存到: {evs_output_dir}")
        return True
        
    except Exception as e:
        print(f"保存EVS txyp间隔数据失败: {e}")
        return False

def save_paired_aps_raw_data(aps_data_dict, output_dir, max_count=10):
    """
    保存配对数据中指定数量的APS为10bit RAW数据
    
    :param aps_data_dict: APS数据字典 {帧号: APS数据}
    :param output_dir: 输出目录
    :param max_count: 最大保存数量
    """
    try:
        # 创建APS RAW数据输出目录
        aps_output_dir = os.path.join(output_dir, "aps_10bit_raw")
        os.makedirs(aps_output_dir, exist_ok=True)
        
        # 获取所有APS帧号并排序
        aps_frame_numbers = sorted(aps_data_dict.keys())
        
        # 只保存最后max_count个APS数据
        if len(aps_frame_numbers) > max_count:
            aps_frame_numbers = aps_frame_numbers[-max_count:]
        
        # 保存APS RAW数据并记录时间戳信息
        saved_count = 0
        saved_aps_info = []  # 保存每个RAW文件的详细信息
        
        for frame_number in aps_frame_numbers:
            aps_data = aps_data_dict[frame_number]
            output_path = os.path.join(aps_output_dir, f"aps_frame_{frame_number:03d}.raw")
            
            if save_aps_as_10bit_raw(aps_data, output_path):
                saved_count += 1
                
                # 获取时间戳信息
                timestamp_us = extract_timestamp_from_data(aps_data)
                timestamp_sec = timestamp_us / 1000000.0 if timestamp_us is not None else None
                
                saved_aps_info.append({
                    'frame_number': frame_number,
                    'filename': f"aps_frame_{frame_number:03d}.raw",
                    'timestamp_us': timestamp_us,
                    'timestamp_sec': timestamp_sec
                })
                
                print(f"保存APS RAW数据: 帧{frame_number}, 时间戳: {timestamp_us} μs")
        
        # 保存APS数据元数据
        metadata_path = os.path.join(aps_output_dir, "metadata.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=== APS 10bit RAW数据提取结果 ===\n")
            f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总APS帧数: {len(aps_data_dict)}\n")
            f.write(f"保存的APS帧数: {saved_count}\n")
            f.write(f"最大保存数量: {max_count}\n")
            
            # 添加分辨率和时间戳信息
            if aps_data_dict:
                first_frame_number = sorted(aps_data_dict.keys())[0]
                first_aps_data = aps_data_dict[first_frame_number]
                
                # 获取分辨率信息
                try:
                    aps_image = first_aps_data.convertTo()
                    if hasattr(aps_image, 'shape'):
                        height, width = aps_image.shape[:2]
                        f.write(f"APS分辨率: {width}x{height}\n")
                    else:
                        f.write("APS分辨率: N/A\n")
                except Exception as e:
                    f.write("APS分辨率: N/A\n")
                
                # 获取时间戳信息
                try:
                    timestamp_us = extract_timestamp_from_data(first_aps_data)
                    if timestamp_us is not None:
                        f.write(f"第一帧时间戳: {timestamp_us} μs\n")
                        f.write(f"第一帧时间戳(秒): {timestamp_us/1000000:.6f} 秒\n")
                    else:
                        f.write("第一帧时间戳: N/A\n")
                except Exception as e:
                    f.write("第一帧时间戳: N/A\n")
                
                # 获取时间戳范围
                try:
                    timestamps = []
                    for frame_number, aps_data in aps_data_dict.items():
                        ts = extract_timestamp_from_data(aps_data)
                        if ts is not None:
                            timestamps.append(ts)
                    
                    if timestamps:
                        min_ts = min(timestamps)
                        max_ts = max(timestamps)
                        f.write(f"时间戳范围: {min_ts}-{max_ts} μs\n")
                        f.write(f"时间戳范围(秒): {min_ts/1000000:.6f}-{max_ts/1000000:.6f} 秒\n")
                        f.write(f"采集持续时间: {(max_ts-min_ts)/1000000:.6f} 秒\n")
                    else:
                        f.write("时间戳范围: N/A\n")
                except Exception as e:
                    f.write("时间戳范围: N/A\n")
            
            f.write("数据格式: 10bit RAW二进制文件\n")
            f.write("文件命名规则: aps_frame_XXX.raw (XXX=帧号)\n")
            f.write("数据类型: uint16 (低10位有效)\n")
            
            # 添加所有保存的RAW文件时间戳详细信息
            if saved_aps_info:
                f.write("\n=== 保存的RAW文件详细时间戳信息 ===\n")
                f.write("帧号\t文件名\t\t\t时间戳(μs)\t\t时间戳(秒)\n")
                f.write("-" * 70 + "\n")
                
                for aps_info in saved_aps_info:
                    frame_num = aps_info['frame_number']
                    filename = aps_info['filename']
                    timestamp_us = aps_info['timestamp_us']
                    timestamp_sec = aps_info['timestamp_sec']
                    
                    if timestamp_us is not None:
                        f.write(f"{frame_num:03d}\t{filename}\t\t{timestamp_us}\t\t{timestamp_sec:.6f}\n")
                    else:
                        f.write(f"{frame_num:03d}\t{filename}\t\tN/A\t\tN/A\n")
        
        print(f"APS 10bit RAW数据已保存到: {aps_output_dir}")
        return True
        
    except Exception as e:
        print(f"保存APS RAW数据失败: {e}")
        return False

def save_paired_raw_and_evs_data(raw_aps_data, raw_evs_data, aps_timestamps, output_dir, max_count=10):
    """
    保存配对数据的APS RAW和EVS txyp数据
    
    :param raw_aps_data: 原始APS数据字典
    :param raw_evs_data: 原始EVS数据字典
    :param aps_timestamps: APS时间戳列表
    :param output_dir: 输出目录
    :param max_count: 最大保存数量
    """
    print("\n=== 开始保存配对RAW和EVS数据 ===")
    
    # 保存APS RAW数据
    print("保存APS 10bit RAW数据...")
    aps_success = save_paired_aps_raw_data(raw_aps_data, output_dir, max_count)
    
    # 保存EVS txyp数据
    print("保存EVS txyp数据...")
    evs_success = save_evs_txyp_intervals(raw_evs_data, output_dir, aps_timestamps, max_count)
    
    if aps_success and evs_success:
        print("配对RAW和EVS数据保存完成！")
    else:
        print("配对RAW和EVS数据保存部分失败！")

def main():
    """
    主函数：读取指定的数据文件并提取机械臂运动期间的APS和EVS数据
    支持单文件和配对采集会话两种模式
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
    data_info = find_data_files(folder_path)
    
    if data_info is None:
        print("未找到数据文件，程序退出")
        return
    
    # 判断数据模式并处理
    if isinstance(data_info, dict) and 'collection_01' in data_info:
        # 配对采集会话模式
        print("\n检测到配对采集会话，开始分析...")
        extract_paired_collection_data(data_info)
    elif isinstance(data_info, dict) and data_info.get('mode') == 'single':
        # 单文件模式
        print("\n检测到单文件数据，开始分析...")
        data_file_path = data_info['data_file']
        sync_timestamps_path = data_info['sync_timestamps']
        
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
    else:
        print("未知的数据格式，程序退出")

if __name__ == "__main__":
    main()