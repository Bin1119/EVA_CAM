# -*- coding: utf-8 -*-
"""
AVI to GIF Converter
将AVI视频文件转换为GIF动画
"""

import cv2
import numpy as np
import os
from PIL import Image
import argparse
from tqdm import tqdm

def avi_to_gif(input_path, output_path=None, fps=10, scale=1.0, quality=85, speed=1.0):
    """
    将AVI文件转换为GIF
    
    Args:
        input_path (str): 输入AVI文件路径
        output_path (str): 输出GIF文件路径（可选）
        fps (int): GIF的帧率
        scale (float): 缩放比例 (0.0-1.0)
        quality (int): GIF质量 (1-100)
        speed (float): 播放速度倍数 (0.1-10.0)
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在: {input_path}")
        return False
    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base_name}.gif")
    
    # 计算实际的FPS（考虑速度倍数）
    actual_fps = fps * speed
    
    print(f"正在转换: {input_path} -> {output_path}")
    print(f"参数: FPS={fps}, 速度={speed}x, 实际FPS={actual_fps}, 缩放={scale}, 质量={quality}")
    
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {input_path}")
            return False
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height}, {original_fps:.2f}fps, {frame_count}帧")
        
        # 计算新的尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 计算采样间隔
        sample_interval = max(1, int(original_fps / actual_fps))
        
        frames = []
        frame_idx = 0
        
        # 读取帧
        with tqdm(total=frame_count, desc="读取视频帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按指定间隔采样
                if frame_idx % sample_interval == 0:
                    # 调整大小
                    if scale != 1.0:
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # 转换颜色空间 BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 转换为PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        if not frames:
            print("错误：没有读取到任何帧")
            return False
        
        print(f"采样了 {len(frames)} 帧用于GIF")
        
        # 保存GIF
        print("正在保存GIF...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // actual_fps,  # 每帧持续时间（毫秒）
            loop=0,  # 无限循环
            optimize=True,
            quality=quality
        )
        
        # 检查输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"转换成功！")
            print(f"输出文件: {output_path}")
            print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
            print(f"GIF尺寸: {new_width}x{new_height}")
            print(f"GIF帧数: {len(frames)}")
            print(f"播放速度: {speed}x")
            return True
        else:
            print("错误：GIF文件未创建")
            return False
            
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将AVI视频文件转换为GIF")
    parser.add_argument("input", help="输入AVI文件路径")
    parser.add_argument("-o", "--output", help="输出GIF文件路径（可选）")
    parser.add_argument("-f", "--fps", type=int, default=10, help="GIF帧率（默认: 10）")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="缩放比例（默认: 1.0）")
    parser.add_argument("-q", "--quality", type=int, default=85, help="GIF质量 1-100（默认: 85）")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍数（默认: 1.0）")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.fps <= 0:
        print("错误：FPS必须大于0")
        return
    
    if args.scale <= 0 or args.scale > 1:
        print("错误：缩放比例必须在0-1之间")
        return
    
    if args.quality < 1 or args.quality > 100:
        print("错误：质量必须在1-100之间")
        return
    
    if args.speed <= 0 or args.speed > 10:
        print("错误：速度倍数必须在0.1-10之间")
        return
    
    # 执行转换
    success = avi_to_gif(
        args.input,
        args.output,
        args.fps,
        args.scale,
        args.quality,
        args.speed
    )
    
    if success:
        print("转换完成！")
    else:
        print("转换失败！")

if __name__ == "__main__":
    main()