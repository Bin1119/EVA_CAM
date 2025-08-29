# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2 as cv
from datetime import datetime
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

def create_bayer_pattern(width, height, pattern='RGGB'):
    """
    创建拜耳阵列模式
    
    :param width: 图像宽度
    :param height: 图像高度
    :param pattern: 拜耳模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
    :return: 拜耳模式数组
    """
    patterns = {
        'RGGB': np.array([[0, 1], [2, 3]]),  # R G, G B
        'BGGR': np.array([[3, 2], [1, 0]]),  # B G, G R
        'GRBG': np.array([[1, 0], [3, 2]]),  # G R, B G
        'GBRG': np.array([[2, 3], [0, 1]]),  # G B, R G
    }
    
    if pattern not in patterns:
        pattern = 'RGGB'
    
    # 创建基础模式
    base_pattern = patterns[pattern]
    
    # 扩展到全图像尺寸
    pattern_array = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            pattern_array[i, j] = base_pattern[i % 2, j % 2]
    
    return pattern_array

def black_level_correction(raw_image, black_level=64):
    """
    黑电平校正
    
    :param raw_image: 原始RAW图像
    :param black_level: 黑电平值
    :return: 校正后的图像
    """
    corrected = raw_image.astype(np.float32) - black_level
    corrected = np.maximum(corrected, 0)  # 确保非负
    return corrected

def white_balance(raw_image, pattern_array, method='gray_world'):
    """
    白平衡处理
    
    :param raw_image: 原始RAW图像
    :param pattern_array: 拜耳模式数组
    :param method: 白平衡方法 ('gray_world', 'perfect_reflector')
    :return: 白平衡增益, 白平衡后的图像
    """
    # 分离各通道像素
    r_pixels = raw_image[pattern_array == 0]
    gr_pixels = raw_image[pattern_array == 1]
    gb_pixels = raw_image[pattern_array == 2]
    b_pixels = raw_image[pattern_array == 3]
    
    # 计算各通道平均值
    r_avg = np.mean(r_pixels)
    g_avg = (np.mean(gr_pixels) + np.mean(gb_pixels)) / 2
    b_avg = np.mean(b_pixels)
    
    if method == 'gray_world':
        # 灰度世界假设
        target_avg = (r_avg + g_avg + b_avg) / 3
        r_gain = target_avg / r_avg if r_avg > 0 else 1.0
        g_gain = target_avg / g_avg if g_avg > 0 else 1.0
        b_gain = target_avg / b_avg if b_avg > 0 else 1.0
    else:
        # 完美反射体假设
        max_val = max(r_avg, g_avg, b_avg)
        r_gain = max_val / r_avg if r_avg > 0 else 1.0
        g_gain = max_val / g_avg if g_avg > 0 else 1.0
        b_gain = max_val / b_avg if b_avg > 0 else 1.0
    
    # 限制增益范围
    r_gain = np.clip(r_gain, 0.5, 4.0)
    g_gain = np.clip(g_gain, 0.5, 4.0)
    b_gain = np.clip(b_gain, 0.5, 4.0)
    
    # 应用白平衡
    balanced_image = raw_image.astype(np.float32).copy()
    balanced_image[pattern_array == 0] *= r_gain
    balanced_image[pattern_array == 1] *= g_gain
    balanced_image[pattern_array == 2] *= g_gain
    balanced_image[pattern_array == 3] *= b_gain
    
    return (r_gain, g_gain, b_gain), balanced_image

def demosaic(balanced_image, pattern_array, method='bilinear'):
    """
    去马赛克处理
    
    :param balanced_image: 白平衡后的图像
    :param pattern_array: 拜耳模式数组
    :param method: 去马赛克方法 ('bilinear', 'malvar', 'edge_directed')
    :return: RGB图像
    """
    height, width = balanced_image.shape
    
    if method == 'bilinear':
        # 双线性插值去马赛克
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # 创建各通道的掩码
        r_mask = (pattern_array == 0)
        g_mask = (pattern_array == 1) | (pattern_array == 2)
        b_mask = (pattern_array == 3)
        
        # 提取各通道的已知像素
        r_known = balanced_image * r_mask
        g_known = balanced_image * g_mask
        b_known = balanced_image * b_mask
        
        # 对每个通道进行插值
        from scipy import ndimage
        
        # R通道
        r_pixels = r_known[r_mask]
        if len(r_pixels) > 0:
            r_interp = ndimage.zoom(r_known.reshape(height//2, width//2), 2, order=1)
            rgb_image[:, :, 0] = r_interp
        
        # G通道
        g_interp = ndimage.zoom(g_known, 0.5, order=1)  # 下采样
        g_interp = ndimage.zoom(g_interp, 2, order=1)    # 上采样
        rgb_image[:, :, 1] = g_interp
        
        # B通道
        b_pixels = b_known[b_mask]
        if len(b_pixels) > 0:
            b_interp = ndimage.zoom(b_known.reshape(height//2, width//2), 2, order=1)
            rgb_image[:, :, 2] = b_interp
        
        return rgb_image
    
    elif method == 'malvar':
        # Malvar算法去马赛克（更高质量）
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # 创建各通道的掩码
        r_mask = (pattern_array == 0)
        g_mask = (pattern_array == 1) | (pattern_array == 2)
        b_mask = (pattern_array == 3)
        
        # 初始化已知像素
        rgb_image[r_mask, 0] = balanced_image[r_mask]
        rgb_image[g_mask, 1] = balanced_image[g_mask]
        rgb_image[b_mask, 2] = balanced_image[b_mask]
        
        # Malvar算法插值
        for c in range(3):
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if rgb_image[i, j, c] == 0:  # 需要插值的像素
                        if c == 0:  # R通道
                            # 绿色像素位置的红色插值
                            if pattern_array[i, j] in [1, 2]:
                                rgb_image[i, j, c] = (balanced_image[i-1, j] + balanced_image[i+1, j] + 
                                                     balanced_image[i, j-1] + balanced_image[i, j+1]) / 4
                            # 蓝色像素位置的红色插值
                            elif pattern_array[i, j] == 3:
                                rgb_image[i, j, c] = (balanced_image[i-1, j-1] + balanced_image[i-1, j+1] + 
                                                     balanced_image[i+1, j-1] + balanced_image[i+1, j+1]) / 4
                        
                        elif c == 1:  # G通道
                            # 红色像素位置的绿色插值
                            if pattern_array[i, j] == 0:
                                rgb_image[i, j, c] = (balanced_image[i-1, j] + balanced_image[i+1, j] + 
                                                     balanced_image[i, j-1] + balanced_image[i, j+1]) / 4
                            # 蓝色像素位置的绿色插值
                            elif pattern_array[i, j] == 3:
                                rgb_image[i, j, c] = (balanced_image[i-1, j] + balanced_image[i+1, j] + 
                                                     balanced_image[i, j-1] + balanced_image[i, j+1]) / 4
                        
                        elif c == 2:  # B通道
                            # 绿色像素位置的蓝色插值
                            if pattern_array[i, j] in [1, 2]:
                                rgb_image[i, j, c] = (balanced_image[i-1, j] + balanced_image[i+1, j] + 
                                                     balanced_image[i, j-1] + balanced_image[i, j+1]) / 4
                            # 红色像素位置的蓝色插值
                            elif pattern_array[i, j] == 0:
                                rgb_image[i, j, c] = (balanced_image[i-1, j-1] + balanced_image[i-1, j+1] + 
                                                     balanced_image[i+1, j-1] + balanced_image[i+1, j+1]) / 4
        
        return rgb_image
    
    else:
        # 简单复制作为fallback
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        rgb_image[:, :, 0] = balanced_image
        rgb_image[:, :, 1] = balanced_image
        rgb_image[:, :, 2] = balanced_image
        return rgb_image

def color_correction(rgb_image, ccm=None):
    """
    色彩校正
    
    :param rgb_image: RGB图像
    :param ccm: 色彩校正矩阵 (3x3)
    :return: 色彩校正后的图像
    """
    if ccm is None:
        # 默认色彩校正矩阵（可根据实际传感器调整）
        ccm = np.array([
            [1.2, -0.1, -0.1],
            [-0.1, 1.1, 0.0],
            [-0.1, 0.0, 1.1]
        ])
    
    # 应用色彩校正矩阵
    height, width = rgb_image.shape[:2]
    rgb_reshaped = rgb_image.reshape(-1, 3)
    corrected = np.dot(rgb_reshaped, ccm.T)
    corrected = corrected.reshape(height, width, 3)
    
    # 限制范围
    corrected = np.clip(corrected, 0, None)
    
    return corrected

def gamma_correction(rgb_image, gamma=2.2):
    """
    伽马校正
    
    :param rgb_image: RGB图像
    :param gamma: 伽马值
    :return: 伽马校正后的图像
    """
    # 归一化到0-1
    normalized = rgb_image / np.max(rgb_image)
    
    # 应用伽马校正
    gamma_corrected = np.power(normalized, 1.0/gamma)
    
    # 恢复到原始范围
    corrected = gamma_corrected * np.max(rgb_image)
    
    return corrected

def denoise(rgb_image, method='bilateral'):
    """
    去噪处理
    
    :param rgb_image: RGB图像
    :param method: 去噪方法 ('bilateral', 'gaussian', 'median')
    :return: 去噪后的图像
    """
    if method == 'bilateral':
        # 双边滤波
        denoised = cv.bilateralFilter(rgb_image.astype(np.uint8), 9, 75, 75)
    elif method == 'gaussian':
        # 高斯滤波
        denoised = cv.GaussianBlur(rgb_image.astype(np.uint8), (5, 5), 0)
    elif method == 'median':
        # 中值滤波
        denoised = cv.medianBlur(rgb_image.astype(np.uint8), 5)
    else:
        denoised = rgb_image
    
    return denoised.astype(np.float32)

def sharpen_image(rgb_image, amount=1.0):
    """
    图像锐化
    
    :param rgb_image: RGB图像
    :param amount: 锐化强度
    :return: 锐化后的图像
    """
    # 创建锐化核
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]]) * amount
    
    # 调整中心值
    kernel[1, 1] = kernel[1, 1] - 8 * amount + 1
    
    # 应用锐化
    sharpened = cv.filter2D(rgb_image.astype(np.uint8), -1, kernel)
    
    return sharpened.astype(np.float32)

def read_10bit_raw_to_8bit_rgb_isp(raw_file_path, width=640, height=480, bayer_pattern='RGGB', 
                                   black_level=64, denoise_method='bilateral', 
                                   demosaic_method='malvar', gamma=2.2, sharpen_amount=0.5):
    """
    读取10bit RAW文件并使用ISP处理转换为8bit RGB图像
    
    :param raw_file_path: RAW文件路径
    :param width: 图像宽度
    :param height: 图像高度
    :param bayer_pattern: 拜耳模式
    :param black_level: 黑电平
    :param denoise_method: 去噪方法
    :param demosaic_method: 去马赛克方法
    :param gamma: 伽马值
    :param sharpen_amount: 锐化强度
    :return: 8bit RGB图像 (numpy数组)
    """
    try:
        # 读取RAW文件
        with open(raw_file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint16)
        
        # 检查数据大小
        expected_size = width * height
        if len(raw_data) != expected_size:
            print(f"警告: 期望大小 {expected_size}, 实际大小 {len(raw_data)}")
            
            # 尝试自动计算合理的图像尺寸
            if len(raw_data) > expected_size:
                total_pixels = len(raw_data)
                print(f"总像素数: {total_pixels}")
                
                # 尝试常见的图像尺寸比例
                common_ratios = [
                    (640, 480), (1280, 960), (320, 240), (800, 600),
                    (1024, 768), (1280, 720), (1920, 1080), (720, 576),
                    (1280, 1024), (3264, 2448), (3840, 2160)
                ]
                
                # 寻找最接近的尺寸
                best_match = None
                min_diff = float('inf')
                
                for w, h in common_ratios:
                    if w * h == total_pixels:
                        best_match = (w, h)
                        break
                    elif abs(w * h - total_pixels) < min_diff:
                        min_diff = abs(w * h - total_pixels)
                        best_match = (w, h)
                
                if best_match:
                    width, height = best_match
                    print(f"自动检测图像尺寸: {width}x{height}")
                    expected_size = width * height
                
                # 调整数据大小
                if len(raw_data) > expected_size:
                    raw_data = raw_data[:expected_size]
                else:
                    padded_data = np.zeros(expected_size, dtype=np.uint16)
                    padded_data[:len(raw_data)] = raw_data
                    raw_data = padded_data
            else:
                padded_data = np.zeros(expected_size, dtype=np.uint16)
                padded_data[:len(raw_data)] = raw_data
                raw_data = padded_data
        
        # 重塑为2D图像
        raw_image = raw_data.reshape((height, width))
        
        # 提取10bit数据 (掩码 0x3FF)
        raw_10bit = raw_image & 0x3FF
        
        print(f"开始ISP处理...")
        print(f"输入数据范围: {raw_10bit.min()} - {raw_10bit.max()}")
        
        # 1. 黑电平校正
        corrected = black_level_correction(raw_10bit, black_level)
        print(f"黑电平校正后范围: {corrected.min():.1f} - {corrected.max():.1f}")
        
        # 2. 创建拜耳模式
        pattern_array = create_bayer_pattern(width, height, bayer_pattern)
        
        # 3. 白平衡
        wb_gains, balanced = white_balance(corrected, pattern_array, method='gray_world')
        print(f"白平衡增益: R={wb_gains[0]:.2f}, G={wb_gains[1]:.2f}, B={wb_gains[2]:.2f}")
        
        # 4. 去马赛克
        rgb_image = demosaic(balanced, pattern_array, method=demosaic_method)
        print(f"去马赛克完成，RGB图像尺寸: {rgb_image.shape}")
        
        # 5. 色彩校正
        rgb_image = color_correction(rgb_image)
        
        # 6. 伽马校正
        rgb_image = gamma_correction(rgb_image, gamma)
        
        # 7. 去噪
        if denoise_method:
            rgb_image = denoise(rgb_image, method=denoise_method)
        
        # 8. 锐化
        if sharpen_amount > 0:
            rgb_image = sharpen_image(rgb_image, amount=sharpen_amount)
        
        # 9. 归一化到8bit
        rgb_min = rgb_image.min()
        rgb_max = rgb_image.max()
        if rgb_max > rgb_min:
            rgb_8bit = ((rgb_image - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
        else:
            rgb_8bit = np.zeros_like(rgb_image, dtype=np.uint8)
        
        print(f"ISP处理完成，输出范围: {rgb_8bit.min()} - {rgb_8bit.max()}")
        
        return rgb_8bit, width, height
        
    except Exception as e:
        print(f"ISP处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, width, height

def read_10bit_raw_to_8bit_rgb(raw_file_path, width=640, height=480, gamma=0.5):
    """
    读取10bit RAW文件并转换为8bit RGB图像
    
    :param raw_file_path: RAW文件路径
    :param width: 图像宽度
    :param height: 图像高度
    :param gamma: 伽马值，小于1用于提亮，大于1用于变暗
    :return: 8bit RGB图像 (numpy数组)
    """
    try:
        # 读取RAW文件
        with open(raw_file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint16)
        
        # 检查数据大小
        expected_size = width * height
        if len(raw_data) != expected_size:
            print(f"警告: 期望大小 {expected_size}, 实际大小 {len(raw_data)}")
            
            # 尝试自动计算合理的图像尺寸
            if len(raw_data) > expected_size:
                # 计算可能的图像尺寸
                total_pixels = len(raw_data)
                print(f"总像素数: {total_pixels}")
                
                # 尝试常见的图像尺寸比例
                common_ratios = [
                    (640, 480),   # 4:3
                    (1280, 960),  # 4:3
                    (320, 240),   # 4:3
                    (800, 600),   # 4:3
                    (1024, 768),  # 4:3
                    (1280, 720),  # 16:9
                    (1920, 1080), # 16:9
                    (720, 576),   # 5:4
                    (1280, 1024), # 5:4
                    (3264, 2448),
                ]
                
                # 寻找最接近的尺寸
                best_match = None
                min_diff = float('inf')
                
                for w, h in common_ratios:
                    if w * h == total_pixels:
                        best_match = (w, h)
                        break
                    elif abs(w * h - total_pixels) < min_diff:
                        min_diff = abs(w * h - total_pixels)
                        best_match = (w, h)
                
                if best_match:
                    width, height = best_match
                    print(f"自动检测图像尺寸: {width}x{height}")
                    expected_size = width * height
                
                # 如果仍然不匹配，截取数据
                if len(raw_data) > expected_size:
                    raw_data = raw_data[:expected_size]
                else:
                    # 填充数据
                    padded_data = np.zeros(expected_size, dtype=np.uint16)
                    padded_data[:len(raw_data)] = raw_data
                    raw_data = padded_data
            else:
                # 填充数据
                padded_data = np.zeros(expected_size, dtype=np.uint16)
                padded_data[:len(raw_data)] = raw_data
                raw_data = padded_data
        
        # 重塑为2D图像
        raw_image = raw_data.reshape((height, width))
        
        # 提取10bit数据 (掩码 0x3FF)
        raw_10bit = raw_image & 0x3FF
        
        # 转换为8bit (右移2位或线性映射)
        # 方法1: 右移2位 (快速但可能损失精度)
        # image_8bit = (raw_10bit >> 2).astype(np.uint8)
        
        # 方法2: 线性映射到0-255并应用gamma校正进行非线性提亮
        normalized = (raw_10bit.astype(np.float32) / 1023.0)
        # 应用gamma校正：gamma < 1 提亮图像，gamma > 1 变暗图像
        gamma_corrected = np.power(normalized, gamma)
        image_8bit = (gamma_corrected * 255).astype(np.uint8)
        
        # 转换为RGB (复制到三个通道)
        rgb_image = cv.cvtColor(image_8bit, cv.COLOR_GRAY2RGB)
        
        return rgb_image, width, height
        
    except Exception as e:
        print(f"读取RAW文件失败: {e}")
        return None, width, height

def process_raw_files_to_rgb_isp(input_dir, output_dir, width=640, height=480, 
                                 bayer_pattern='RGGB', black_level=64, 
                                 denoise_method='bilateral', demosaic_method='malvar', 
                                 gamma=2.2, sharpen_amount=0.5):
    """
    批量处理RAW文件为RGB图像（使用ISP处理）
    
    :param input_dir: 输入目录 (包含.raw文件)
    :param output_dir: 输出目录 (保存RGB图像)
    :param width: 图像宽度
    :param height: 图像高度
    :param bayer_pattern: 拜耳模式
    :param black_level: 黑电平
    :param denoise_method: 去噪方法
    :param demosaic_method: 去马赛克方法
    :param gamma: 伽马值
    :param sharpen_amount: 锐化强度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有RAW文件
    raw_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.raw'):
            raw_files.append(filename)
    
    if not raw_files:
        print(f"在目录 {input_dir} 中未找到RAW文件")
        return
    
    # 按文件名排序
    raw_files.sort()
    
    print(f"找到 {len(raw_files)} 个RAW文件")
    print(f"ISP参数: 拜耳模式={bayer_pattern}, 黑电平={black_level}, 去噪={denoise_method}")
    print(f"去马赛克方法={demosaic_method}, 伽马={gamma}, 锐化强度={sharpen_amount}")
    
    # 处理每个RAW文件
    success_count = 0
    for filename in raw_files:
        raw_path = os.path.join(input_dir, filename)
        
        # 使用ISP处理转换为RGB
        rgb_image, actual_width, actual_height = read_10bit_raw_to_8bit_rgb_isp(
            raw_path, width, height, bayer_pattern, black_level, 
            denoise_method, demosaic_method, gamma, sharpen_amount)
        
        if rgb_image is not None:
            # 生成输出文件名
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_isp.jpg")
            
            # 保存RGB图像
            cv.imwrite(output_path, rgb_image)
            print(f"ISP转换成功: {filename} -> {os.path.basename(output_path)} (尺寸: {actual_width}x{actual_height})")
            success_count += 1
        else:
            print(f"ISP转换失败: {filename}")
    
    print(f"\nISP转换完成！成功转换 {success_count}/{len(raw_files)} 个文件")
    print(f"RGB图像已保存到: {output_dir}")

def process_raw_files_to_rgb(input_dir, output_dir, width=640, height=480):
    """
    批量处理RAW文件为RGB图像
    
    :param input_dir: 输入目录 (包含.raw文件)
    :param output_dir: 输出目录 (保存RGB图像)
    :param width: 图像宽度
    :param height: 图像高度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有RAW文件
    raw_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.raw'):
            raw_files.append(filename)
    
    if not raw_files:
        print(f"在目录 {input_dir} 中未找到RAW文件")
        return
    
    # 按文件名排序
    raw_files.sort()
    
    print(f"找到 {len(raw_files)} 个RAW文件")
    
    # 处理每个RAW文件
    success_count = 0
    for filename in raw_files:
        raw_path = os.path.join(input_dir, filename)
        
        # 转换为RGB
        rgb_image, actual_width, actual_height = read_10bit_raw_to_8bit_rgb(raw_path, width, height)
        
        if rgb_image is not None:
            # 生成输出文件名
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.jpg")
            
            # 保存RGB图像
            cv.imwrite(output_path, rgb_image)
            print(f"转换成功: {filename} -> {os.path.basename(output_path)} (尺寸: {actual_width}x{actual_height})")
            success_count += 1
        else:
            print(f"转换失败: {filename}")
    
    print(f"\n转换完成！成功转换 {success_count}/{len(raw_files)} 个文件")
    print(f"RGB图像已保存到: {output_dir}")

def display_raw_file(raw_file_path, width=640, height=480):
    """
    显示RAW文件内容
    
    :param raw_file_path: RAW文件路径
    :param width: 图像宽度
    :param height: 图像高度
    """
    # 转换为RGB
    rgb_image, actual_width, actual_height = read_10bit_raw_to_8bit_rgb(raw_file_path, width, height)
    
    if rgb_image is not None:
        # 显示图像
        cv.imshow('RAW to RGB', rgb_image)
        print(f"显示图像: {raw_file_path}")
        print(f"实际尺寸: {actual_width}x{actual_height}")
        print("按任意键关闭窗口...")
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        return True
    else:
        print("无法显示图像")
        return False

def get_raw_file_info(raw_file_path, width=640, height=480):
    """
    获取RAW文件信息
    
    :param raw_file_path: RAW文件路径
    :param width: 图像宽度
    :param height: 图像高度
    """
    try:
        # 读取RAW文件
        with open(raw_file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint16)
        
        print(f"RAW文件信息: {raw_file_path}")
        print(f"文件大小: {len(raw_data)} 像素")
        print(f"期望大小: {width * height} 像素")
        print(f"数据类型: uint16")
        
        if len(raw_data) >= width * height:
            # 重塑为2D图像
            raw_image = raw_data[:width*height].reshape((height, width))
            
            # 提取10bit数据
            raw_10bit = raw_image & 0x3FF
            
            print(f"有效数据范围: {raw_10bit.min()} - {raw_10bit.max()}")
            print(f"数据平均值: {raw_10bit.mean():.2f}")
            print(f"数据标准差: {raw_10bit.std():.2f}")
            
            # 统计数据分布
            hist, bins = np.histogram(raw_10bit, bins=10)
            print("数据分布:")
            for i in range(len(hist)):
                print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}: {hist[i]} 像素")
        
        return True
        
    except Exception as e:
        print(f"获取RAW文件信息失败: {e}")
        return False

def display_raw_file_isp(raw_file_path, width=640, height=480, bayer_pattern='RGGB', 
                         black_level=64, denoise_method='bilateral', 
                         demosaic_method='malvar', gamma=2.2, sharpen_amount=0.5):
    """
    显示RAW文件内容（使用ISP处理）
    
    :param raw_file_path: RAW文件路径
    :param width: 图像宽度
    :param height: 图像高度
    :param bayer_pattern: 拜耳模式
    :param black_level: 黑电平
    :param denoise_method: 去噪方法
    :param demosaic_method: 去马赛克方法
    :param gamma: 伽马值
    :param sharpen_amount: 锐化强度
    """
    # 使用ISP处理转换为RGB
    rgb_image, actual_width, actual_height = read_10bit_raw_to_8bit_rgb_isp(
        raw_file_path, width, height, bayer_pattern, black_level, 
        denoise_method, demosaic_method, gamma, sharpen_amount)
    
    if rgb_image is not None:
        # 显示图像
        cv.imshow('RAW to RGB (ISP)', rgb_image)
        print(f"显示图像: {raw_file_path}")
        print(f"实际尺寸: {actual_width}x{actual_height}")
        print("按任意键关闭窗口...")
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        return True
    else:
        print("无法显示图像")
        return False

def process_collection_aps_to_rgb(base_dir, width=640, height=480, use_isp=True):
    """
    处理collection文件夹下的aps_10bit_raw文件夹，转换为RGB并保存到aps_8bit_rgb文件夹
    
    :param base_dir: 基础目录路径
    :param width: 图像宽度
    :param height: 图像高度
    :param use_isp: 是否使用ISP处理
    """
    print(f"开始处理目录: {base_dir}")
    
    # 检查collection_01_analysis和collection_02_analysis文件夹
    collections = []
    collection_01_dir = os.path.join(base_dir, 'collection_01_analysis')
    collection_02_dir = os.path.join(base_dir, 'collection_02_analysis')
    
    if os.path.exists(collection_01_dir):
        collections.append(('collection_01_analysis', collection_01_dir))
    if os.path.exists(collection_02_dir):
        collections.append(('collection_02_analysis', collection_02_dir))
    
    if not collections:
        print(f"在目录 {base_dir} 中未找到collection_01_analysis或collection_02_analysis文件夹")
        return
    
    # 处理每个collection
    for collection_name, collection_dir in collections:
        print(f"\n处理 {collection_name}...")
        
        # 检查aps_10bit_raw文件夹
        raw_dir = os.path.join(collection_dir, 'aps_10bit_raw')
        if not os.path.exists(raw_dir):
            print(f"未找到 {collection_name}/aps_10bit_raw 文件夹")
            continue
        
        # 创建aps_8bit_rgb输出文件夹
        rgb_dir = os.path.join(collection_dir, 'aps_8bit_rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        
        # 查找所有RAW文件
        raw_files = []
        for filename in os.listdir(raw_dir):
            if filename.endswith('.raw'):
                raw_files.append(filename)
        
        if not raw_files:
            print(f"在 {raw_dir} 中未找到RAW文件")
            continue
        
        # 按文件名排序
        raw_files.sort()
        
        print(f"找到 {len(raw_files)} 个RAW文件")
        
        # 处理每个RAW文件
        success_count = 0
        for filename in raw_files:
            raw_path = os.path.join(raw_dir, filename)
            
            if use_isp:
                # 使用ISP处理转换为RGB
                rgb_image, actual_width, actual_height = read_10bit_raw_to_8bit_rgb_isp(
                    raw_path, width, height)
                output_filename = f"{os.path.splitext(filename)[0]}_isp.jpg"
            else:
                # 使用传统方法转换为RGB
                rgb_image, actual_width, actual_height = read_10bit_raw_to_8bit_rgb(
                    raw_path, width, height)
                output_filename = f"{os.path.splitext(filename)[0]}.jpg"
            
            if rgb_image is not None:
                # 保存RGB图像
                output_path = os.path.join(rgb_dir, output_filename)
                cv.imwrite(output_path, rgb_image)
                print(f"转换成功: {filename} -> {output_filename} (尺寸: {actual_width}x{actual_height})")
                success_count += 1
            else:
                print(f"转换失败: {filename}")
        
        print(f"{collection_name} 转换完成！成功转换 {success_count}/{len(raw_files)} 个文件")
        print(f"RGB图像已保存到: {rgb_dir}")
    
    print(f"\n所有处理完成！")

def main():
    """
    主函数：处理RAW文件转换为RGB图像
    """
    print("=== RAW文件转RGB工具 ===")
    print("1. 自动处理collection文件夹下的APS RAW文件")
    print("2. 批量转换RAW文件为RGB图像（传统方法）")
    print("3. 批量转换RAW文件为RGB图像（ISP处理）")
    print("4. 显示单个RAW文件（传统方法）")
    print("5. 显示单个RAW文件（ISP处理）")
    print("6. 查看RAW文件信息")
    print("7. 对比显示同一文件的两种处理方法")
    
    choice = input("请选择功能 (1/2/3/4/5/6/7): ").strip()
    
    if choice == "1":
        # 自动处理collection文件夹
        base_dir = input("请输入包含collection_01_analysis和collection_02_analysis的基础目录路径: ").strip()
        
        # 选择处理方法
        method_choice = input("请选择处理方法 (1: 传统方法, 2: ISP处理): ").strip()
        use_isp = (method_choice == "2")
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        process_collection_aps_to_rgb(base_dir, width, height, use_isp)
        
    elif choice == "2":
        # 批量转换（传统方法）
        input_dir = input("请输入RAW文件所在目录路径: ").strip()
        output_dir = input("请输入RGB图像输出目录路径: ").strip()
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        process_raw_files_to_rgb(input_dir, output_dir, width, height)
        
    elif choice == "3":
        # 批量转换（ISP处理）
        input_dir = input("请输入RAW文件所在目录路径: ").strip()
        output_dir = input("请输入RGB图像输出目录路径: ").strip()
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        # ISP参数设置
        use_custom_isp = input("是否自定义ISP参数? (y/n): ").strip().lower()
        if use_custom_isp == 'y':
            bayer_pattern = input("请输入拜耳模式 (RGGB/BGGR/GRBG/GBRG): ").strip().upper()
            black_level = int(input("请输入黑电平 (默认64): ") or "64")
            denoise_method = input("请输入去噪方法 (bilateral/gaussian/median/none): ").strip().lower()
            if denoise_method == 'none':
                denoise_method = None
            demosaic_method = input("请输入去马赛克方法 (bilinear/malvar): ").strip().lower()
            gamma = float(input("请输入伽马值 (默认2.2): ") or "2.2")
            sharpen_amount = float(input("请输入锐化强度 (0-1, 默认0.5): ") or "0.5")
        else:
            bayer_pattern, black_level, denoise_method, demosaic_method, gamma, sharpen_amount = \
                'RGGB', 64, 'bilateral', 'malvar', 2.2, 0.5
        
        process_raw_files_to_rgb_isp(input_dir, output_dir, width, height, 
                                   bayer_pattern, black_level, denoise_method, 
                                   demosaic_method, gamma, sharpen_amount)
        
    elif choice == "4":
        # 显示单个文件（传统方法）
        raw_file_path = input("请输入RAW文件路径: ").strip()
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        display_raw_file(raw_file_path, width, height)
        
    elif choice == "5":
        # 显示单个文件（ISP处理）
        raw_file_path = input("请输入RAW文件路径: ").strip()
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        # ISP参数设置
        use_custom_isp = input("是否自定义ISP参数? (y/n): ").strip().lower()
        if use_custom_isp == 'y':
            bayer_pattern = input("请输入拜耳模式 (RGGB/BGGR/GRBG/GBRG): ").strip().upper()
            black_level = int(input("请输入黑电平 (默认64): ") or "64")
            denoise_method = input("请输入去噪方法 (bilateral/gaussian/median/none): ").strip().lower()
            if denoise_method == 'none':
                denoise_method = None
            demosaic_method = input("请输入去马赛克方法 (bilinear/malvar): ").strip().lower()
            gamma = float(input("请输入伽马值 (默认2.2): ") or "2.2")
            sharpen_amount = float(input("请输入锐化强度 (0-1, 默认0.5): ") or "0.5")
        else:
            bayer_pattern, black_level, denoise_method, demosaic_method, gamma, sharpen_amount = \
                'RGGB', 64, 'bilateral', 'malvar', 2.2, 0.5
        
        display_raw_file_isp(raw_file_path, width, height, bayer_pattern, black_level, 
                           denoise_method, demosaic_method, gamma, sharpen_amount)
        
    elif choice == "6":
        # 查看文件信息
        raw_file_path = input("请输入RAW文件路径: ").strip()
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        get_raw_file_info(raw_file_path, width, height)
        
    elif choice == "7":
        # 对比显示
        raw_file_path = input("请输入RAW文件路径: ").strip()
        
        # 可选：自定义图像尺寸
        use_custom_size = input("是否自定义图像尺寸? (y/n): ").strip().lower()
        if use_custom_size == 'y':
            width = int(input("请输入图像宽度: "))
            height = int(input("请输入图像高度: "))
        else:
            width, height = 640, 480  # 默认尺寸
        
        # 获取传统方法处理的图像
        rgb_traditional, w1, h1 = read_10bit_raw_to_8bit_rgb(raw_file_path, width, height)
        
        # 获取ISP处理的图像
        rgb_isp, w2, h2 = read_10bit_raw_to_8bit_rgb_isp(raw_file_path, width, height)
        
        if rgb_traditional is not None and rgb_isp is not None:
            # 创建对比图像
            h, w = rgb_traditional.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            
            # 放置传统方法图像
            comparison[:, :w, :] = rgb_traditional
            
            # 放置ISP处理图像
            comparison[:, w:, :] = rgb_isp
            
            # 添加标签
            cv.putText(comparison, "Traditional Method", (10, 30), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(comparison, "ISP Processing", (w + 10, 30), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示对比图像
            cv.imshow('RAW to RGB Comparison', comparison)
            print(f"对比显示: {raw_file_path}")
            print("左侧：传统方法（灰度复制）")
            print("右侧：ISP处理（完整色彩还原）")
            print("按任意键关闭窗口...")
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("无法生成对比图像")
        
    else:
        print("无效选择")

if __name__ == "__main__":
    main()