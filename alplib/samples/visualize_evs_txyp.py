#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVS TXYP数据可视化工具
读取txyp格式的EVS数据并投影到图像上
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path

def visualize_evs_txyp(txyp_file, output_width=816, output_height=612):
    """
    将EVS TXYP数据可视化投影到图像上
    
    Args:
        txyp_file: txyp格式的npy文件路径
        output_width: 输出图像宽度
        output_height: 输出图像高度
    
    Returns:
        visualization_img: 可视化图像
    """
    try:
        # 读取txyp数据
        print(f"正在读取文件: {txyp_file}")
        txyp_data = np.load(txyp_file)
        
        print(f"数据形状: {txyp_data.shape}")
        print(f"数据类型: {txyp_data.dtype}")
        
        if txyp_data.shape[1] != 4:
            raise ValueError(f"TXYP数据应该有4列(t,x,y,p)，但实际有{txyp_data.shape[1]}列")
        
        # 提取各列数据
        t = txyp_data[:, 0]  # 时间戳
        x = txyp_data[:, 1]  # x坐标
        y = txyp_data[:, 2]  # y坐标
        p = txyp_data[:, 3]  # 极性(0或1)
        
        print(f"时间戳范围: {t.min():.3f} - {t.max():.3f}")
        print(f"X坐标范围: {x.min()} - {x.max()}")
        print(f"Y坐标范围: {y.min()} - {y.max()}")
        print(f"极性分布: p=0: {(p==0).sum()}, p=1: {(p==1).sum()}")
        
        # 创建白色背景图像
        visualization_img = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
        
        # 确保坐标在图像范围内
        x_valid = np.clip(x, 0, output_width - 1).astype(int)
        y_valid = np.clip(y, 0, output_height - 1).astype(int)
        
        # 分离正负事件
        positive_events = p == 1
        negative_events = p == 0
        
        # 绘制正事件（红色）
        if np.any(positive_events):
            pos_x = x_valid[positive_events]
            pos_y = y_valid[positive_events]
            visualization_img[pos_y, pos_x] = [0, 0, 255]  # BGR格式：红色
        
        # 绘制负事件（蓝色）
        if np.any(negative_events):
            neg_x = x_valid[negative_events]
            neg_y = y_valid[negative_events]
            visualization_img[neg_y, neg_x] = [255, 0, 0]  # BGR格式：蓝色
        
        print(f"可视化完成，图像尺寸: {visualization_img.shape}")
        
        return visualization_img
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

def save_visualization(visualization_img, output_path):
    """保存可视化结果"""
    try:
        cv2.imwrite(output_path, visualization_img)
        print(f"可视化结果已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"保存图像时出错: {e}")
        return False

def main():
    """主函数"""
    # 文件路径
    txyp_file = r"D:\Codes\EVA_CAM\eva_cam\data\paired_collection_session_20250829155855773\collection_02_analysis\evs_txyp_data\evs_491_531.npy"
    
    # 检查文件是否存在
    if not os.path.exists(txyp_file):
        print(f"错误: 文件不存在: {txyp_file}")
        return
    
    # 创建输出目录
    output_dir = Path(txyp_file).parent / "visualization"
    output_dir.mkdir(exist_ok=True)
    
    # 设置输出图像尺寸
    output_width = 816
    output_height = 612
    
    print(f"EVS TXYP数据可视化工具")
    print(f"输入文件: {txyp_file}")
    print(f"输出尺寸: {output_width}x{output_height}")
    print("=" * 50)
    
    # 生成可视化
    visualization_img = visualize_evs_txyp(txyp_file, output_width, output_height)
    
    if visualization_img is not None:
        # 保存结果
        output_file = output_dir / "evs_visualization_816x612.png"
        if save_visualization(visualization_img, str(output_file)):
            print("\n可视化完成！")
            print(f"结果保存在: {output_file}")
            
            # 可选：显示图像
            try:
                cv2.imshow("EVS TXYP Visualization", visualization_img)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"显示图像时出错: {e}")
    else:
        print("可视化失败")

if __name__ == "__main__":
    main()