# -*- coding: utf-8 -*-

##
 # @brief 播放器使用流程说明
 #
 # 本类提供播放器的管理与操作，包括初始化、加载数据、启动播放、监控设备状态、释放资源。
 # 以下是播放器的完整使用步骤：
 #
 # ##一.  播放器实例化##：
 #    - 创建播放器实例，准备后续操作。
 #
 # ##二.  初始化播放器##：
 #    - 调用 `init()` 方法初始化播放器。
 #    - 根据需求选择 APS 模式、EVS 模式，或同时启用两者。
 #    - 确保播放器在初始化时被正确配置，进入工作状态。
 #
 # ##三.  添加数据回调##：
 #    - 3.1 获取 APS 数据: 调用 `getApsFrames()` 获取回调的 APS 数据。
 #    - 3.2 获取 EVS 数据: 调用 `getEvsFrames()` 获取回调的 EVS 数据。
 #    - 3.3 获取同步数据: 调用 `getSyncFrames()` 获取回调的同步数据。
 #
 # ##四.  加载数据##：
 #    - 调用 `load()` 方法加载数据文件并查找数据帧。
 #    - 数据加载完成后，准备开始播放。
 #
 # ##五.  播放数据流##：
 #    - 使用 `play()` 方法启动数据流播放。
 #    - 设备进入数据采集模式，开始接收 APS 或 EVS 数据。
 #
 # ##六.  监控设备状态##：
 #    - 调用 `()` 方法持续监控设备状态。
 #    - 检查设备运行情况，确保数据流稳定并没有异常。
 #
 # ##七.  关闭播放器##：
 #    - 调用 `close()` 方法停止播放器，释放资源。
 #
 # 以上步骤展示了播放器的完整操作流程，按此顺序执行可确保设备正常运行与关闭。
##
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

def apsFrameCallback(player):
    """
    该函数用于处理从设备接收到 APS 的数据帧

    :param player: 播放器实例
    """

    cv.namedWindow("APS", cv.WINDOW_NORMAL)

    ##六.  监控设备状态##
    while player.isWorking():

        # 3.1 获取 APS 数据
        frameslists = player.getApsFrames()

        for it in frameslists:

            # 转换为 OpenCV Mat 数据
            aps_image = it.convertTo()
            cv.imshow("APS", aps_image)
            cv.waitKey(1)
            time.sleep(1 / 30)

    cv.destroyWindow("APS")


def evsFrameCallback(player):
    """
    该函数用于处理从设备接收到 EVS 的数据帧

    :param player: 播放器实例
    """

    cv.namedWindow("EVS", cv.WINDOW_NORMAL)

    ##六.  监控设备状态##
    while player.isWorking():

        # 3.2 获取 EVS 数据
        frameslists = player.getEvsFrames()

        for it in frameslists:
            # 转换为 OpenCV Mat 数据
            evs_image = it.frame()
            # 放大图像数值以便于观察
            cv.imshow("EVS", evs_image * 100)
            cv.waitKey(1)
            time.sleep(1 / 30)

    cv.destroyWindow("EVS")


def syncFrameCallback(player, video_writer=None):
    """
    该函数用于处理从设备接收到同步数据帧

    :param device: 设备实例
    :param video_writer: OpenCV VideoWriter对象用于保存视频
    """
    cv.namedWindow("Sync", cv.WINDOW_NORMAL)

    ##六.  监控设备状态##
    while player.isWorking():

        # 3.3 获取同步数据
        sync_list = player.getSyncFrames()

        for it in sync_list:
            aps_image = it[0].convertTo()
            if len(it[1]) == 0:
                continue 
            for evs in it[1]:
                evs_image = evs.frame()   
                aps_image_resized = cv.resize(aps_image, (evs_image.shape[1], evs_image.shape[0]), interpolation=cv.INTER_LINEAR)
                result = cv.hconcat([aps_image_resized, evs_image * 100])
                cv.imshow("Sync", result)
                
                # 保存视频帧
                if video_writer is not None:
                    video_writer.write(result)
                
                cv.waitKey(1)
                time.sleep(1 / 30)

    cv.destroyWindow("Sync")


def readApsEvsInfo(path):
    """
    此函数为读取ApsEvsInfo.txt 内容

    :param path: ApsEvsInfo.txt文件路径

    :return: 包含三个元素的元组 (success, aps, evs)，分别表示 True 成功 False 失败、APS bin 文件路径和EVS bin 文件路径
    """

    aps = ""
    evs = ""
    success = False

    try:
        with open(path, 'r') as file:
            bin_path = path
            to_remove = "ApsEvsInfo.txt"
            pos = bin_path.find(to_remove)
            if pos != -1:
                bin_path = bin_path[:pos]

            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()

                # 忽略空行
                if not line:
                    continue

                if "APS/" in line:
                    aps = os.path.join(bin_path, line)
                    success = True
                elif "EVS/" in line:
                    evs = os.path.join(bin_path, line)
                    success = True

    except FileNotFoundError:
        print(f"Open file failure: {path}")
        return False

    if not success:
        print("No valid APS or EVS entries found in the file.")
    
    return success, aps, evs



def configurePlayBinData(path):

    """
    配置播放数据的属性

    :param path: 数据文件的路径
    :return: 配置好的播放数据属性
    """
    play_bin_atrr = PlayBinDataAttr()
    play_bin_atrr.pathanme = path
    play_bin_atrr.data_type = "ALPIX_V2"
    play_bin_atrr.width = 0
    play_bin_atrr.height = 0
    return play_bin_atrr


def loadData(player,path, model=3):
    """
    加载保存的 bin 和 hdf5 数据

    :param player: 播放器实例
    :param path: 播放文件路径
    :param model: 数据模式 (1=APS, 2=EVS, 3=HVS/同步)
    """

    if ".bin" in path:
        if "APS" in path:
            player_aps_attr = configurePlayBinData(path)
            ##二.  初始化播放器##
            return player.init(player_aps_attr, None), 1
        elif "EVS" in path:
            player_evs_attr = configurePlayBinData(path)
            ##二.  初始化播放器##
            return player.init(None, player_evs_attr), 2

    # 只有在没有指定model时才询问用户
    if model == 0:
        print("Please choose the data type to load")
        print("Enter 'APS' for APS")
        print("Enter 'EVS' for EVS")
        print("Enter 'HVS' or any key for both")
        print("Please enter the command: ")
        user_input = input()
        if user_input.lower() == 'aps':
            model = 1
        elif user_input.lower() == 'evs':
            model = 2
        else:
            model = 3

    if ".alpdata" in path:
        ##二.  初始化播放器##
        return player.init(PlayAlpDataType(model), path), model

    elif "ApsEvsInfo.txt" in path:
        success, aps, evs = readApsEvsInfo(path)
        if not success:
            return False,model

        player_aps = configurePlayBinData(aps)
        player_evs = configurePlayBinData(evs)

        if model == 1:
            ##二.  初始化播放器##
            return player.init(player_aps, None), model
        elif model == 2:
            ##二.  初始化播放器##
            return player.init(None, player_evs), model
        else:
            ##二.  初始化播放器##
            return player.init(player_aps,player_evs), model

    else:
        return False, model


# SDK 基本调用流程示例，播放数据
def main():

   ##一.  播放器实例化##
    player = AlpPlayer()

    print("Input playback files use absolute paths (*.alpdata or *.ApsEvsInfo.txt or *.bin):")
    play_name = input()

    print("Please choose the data type to load")
    print("Enter 'APS' for APS")
    print("Enter 'EVS' for EVS")
    print("Enter 'HVS' or any key for both")
    print("Please enter the command: ")
    user_input = input()
    model = 0
    if user_input.lower() == 'aps':
        model = 1
    elif user_input.lower() == 'evs':
        model = 2
    else:
        model = 3

    ##二.  初始化播放器##
    success, model = loadData(player,str(play_name), model)

    if not success:
        print("Failed to load data. Exiting...")
        return

    ##三.  添加数据回调##
    if model == 1:
        callback_func = apsFrameCallback
    elif model == 2:
        callback_func = evsFrameCallback
    else:
        callback_func = syncFrameCallback

    # 创建视频写入器（仅在同步模式下）
    video_writer_ref = [None]
    if model == 3:
        # 生成输出视频路径（与输入文件同目录）
        input_path = os.path.dirname(play_name)
        input_filename = os.path.splitext(os.path.basename(play_name))[0]
        output_video_path = os.path.join(input_path, f"{input_filename}_sync.avi")
        
        # 设置视频编码器和参数
        # 使用更通用的编码器
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 30
        print(f"Video codec: XVID, FPS: {fps}")
        # 创建一个标志来指示是否需要初始化VideoWriter
        video_writer_initialized = [False]
        
        # 修改syncFrameCallback函数以支持延迟初始化
        def syncFrameCallbackWithInit(player, video_writer_ref, initialized_ref):
            """
            包装函数，支持延迟初始化VideoWriter
            """
            cv.namedWindow("Sync", cv.WINDOW_NORMAL)
            
            while player.isWorking():
                sync_list = player.getSyncFrames()
                
                for it in sync_list:
                    aps_image = it[0].convertTo()
                    if len(it[1]) == 0:
                        continue
                    for evs in it[1]:
                        evs_image = evs.frame()
                        aps_image_resized = cv.resize(aps_image, (evs_image.shape[1], evs_image.shape[0]), interpolation=cv.INTER_LINEAR)
                        result = cv.hconcat([aps_image_resized, evs_image * 100])
                        cv.imshow("Sync", result)
                        
                        # 延迟初始化VideoWriter
                        if not initialized_ref[0] and video_writer_ref[0] is None:
                            frame_size = (result.shape[1], result.shape[0])
                            print(f"Frame size: {frame_size}, Result shape: {result.shape}, Result dtype: {result.dtype}")
                            
                            video_writer_ref[0] = cv.VideoWriter(output_video_path, fourcc, fps, frame_size)
                            initialized_ref[0] = True
                            print(f"Saving synchronized video to: {output_video_path}")
                            print(f"VideoWriter opened: {video_writer_ref[0].isOpened()}")
                        
                        # 保存视频帧
                        if initialized_ref[0] and video_writer_ref[0] is not None:
                            if video_writer_ref[0].isOpened():
                                # 创建要写入的帧的副本，避免修改原始显示帧
                                write_frame = result.copy()
                                
                                # 确保图像是3通道BGR格式
                                if len(write_frame.shape) == 2:
                                    write_frame = cv.cvtColor(write_frame, cv.COLOR_GRAY2BGR)
                                elif write_frame.shape[2] == 4:
                                    write_frame = cv.cvtColor(write_frame, cv.COLOR_BGRA2BGR)
                                
                                # 确保图像数据类型正确
                                if write_frame.dtype != np.uint8:
                                    write_frame = write_frame.astype(np.uint8)
                                
                                video_writer_ref[0].write(write_frame)
                                
                                # 添加帧计数器
                                if not hasattr(syncFrameCallbackWithInit, 'frame_count'):
                                    syncFrameCallbackWithInit.frame_count = 0
                                syncFrameCallbackWithInit.frame_count += 1
                                if syncFrameCallbackWithInit.frame_count % 30 == 0:  # 每30帧打印一次
                                    print(f"Written {syncFrameCallbackWithInit.frame_count} frames")
                            else:
                                print("Warning: VideoWriter is not opened properly")
                        
                        cv.waitKey(1)
                        time.sleep(1 / 30)
            
            cv.destroyWindow("Sync")
            
            # 打印总帧数
            if hasattr(syncFrameCallbackWithInit, 'frame_count'):
                print(f"Total frames written: {syncFrameCallbackWithInit.frame_count}")
        
        # 使用包装函数
        callback_func = lambda player: syncFrameCallbackWithInit(player, video_writer_ref, video_writer_initialized)
        
        # 创建并启动线程
        t = threading.Thread(target=callback_func, args=(player,))
        t.start()
    else:
        # 创建并启动线程（非同步模式）
        t = threading.Thread(target=callback_func, args=(player,))
        t.start()

    # 线程已经在上面创建了，这里不需要重复创建

    ##四.  加载数据##
    if not player.load():
        print("Failed to load player. Exiting...")
        return

    ##五.  播放数据流##
    if not player.play():
        print("Failed to start playback. Exiting...")
        return

    ##六.  监控设备状态##
    while player.isWorking():
        pass

    ##七.  关闭播放器##
    player.close()

    t.join()
    
    # 释放视频写入器（仅在同步模式下）
    if model == 3 and video_writer_ref[0] is not None:
        if video_writer_ref[0].isOpened():
            video_writer_ref[0].release()
            print("Video saved successfully!")
            # 检查文件是否成功创建
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                print(f"Video file size: {file_size} bytes")
            else:
                print("Warning: Video file was not created")
        else:
            print("Warning: VideoWriter was not properly opened")

    
if __name__ == "__main__":
    main()
