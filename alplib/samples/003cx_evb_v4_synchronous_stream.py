# -*- coding: utf-8 -*-

##
 # @brief 设备使用基本步骤说明
 #
 # 本类提供对设备的管理和操作，包括初始化、配置、启动数据流、监控设备状态、停止数据流并释放资源。
 # 以下是设备使用的完整步骤：
 #
 # ##一.  设备实例化##：
 #    - 创建设备的实例对象，准备开始对设备的操作。
 #    - 设备实例化后，将用于后续的配置与操作。
 #
 # ##二.  初始化设备##：
 #    - 调用 `init()` 方法对设备进行初始化。
 #    - 在初始化过程中，根据需求选择 APS 模式或 EVS 模式，或者同时启用两者。
 #    - 确保设备在初始化时正确配置，以便进入正确的工作状态。
 #
 # ##三.  选择并配置 EVB 设备##：
 #    - 使用 `getCurrentMidSupportDevices()` 获取支持的 EVB 设备列表。
 #    - 调用 `selectCurrentDevice()` 选择要使用的设备，进行 EVB 设备的配置。
 #    - 配置包括设备的操作模式、数据传输方式等。
 #
 # ##四.  打开设备##：
 #    - 调用 `open()` 方法打开设备，并传入配置文件或其他必要参数。
 #    - 设备成功打开后，确保设备能够正常工作并处于就绪状态。
 #
 # ##五.  添加数据回调##：
 #    - 5.1 开启回调数据流: 调用 `startStream()` 开启回调数据流。
 #    - 5.2 开启数据同步: 调用 `enableSync()` 开启数据同步。
 #    - 5.3 获取同步数据: 调用 `getSyncFrames()` 获取回调的同步数据。
 #    - 5.4 停止回调数据流: 调用 `stopStream()` 停止回调数据流。
 #
 # ##六.  启动数据流##：
 #    - 使用 `start()` 方法启动设备的数据流，开始采集数据。
 #    - 此时，设备进入实时数据采集模式，可以开始获取 APS 或 EVS 数据。
 #
 # ##七.  监控设备状态##：
 #    - 调用 `isOpened()` 方法。
 #    - 持续检查设备的运行状态，确保设备能够正常运行并且数据流稳定。
 #    - 可以调用设备提供的状态检查方法，查看设备是否有异常或错误。
 #
 # ##八.  停止数据流##：
 #    - 在结束操作时，调用 `stop()` 停止设备的数据流。
 #
 # ##九.  关闭设备##：
 #    - 使用 `close()` 方法关闭设备，释放设备占用的资源。
 #
 # 以上步骤是设备使用的完整流程，按照此流程操作能够确保设备正确运行和关闭。
 # 设备从初始化、配置到数据流管理、状态监控，再到停止和资源释放，所有环节都应完整执行。
##
import os
import sys

# 查询目录地址
current_dir = os.path.dirname(os.path.abspath(__file__))
# 查询pyd地址
py3_path = os.path.join(current_dir, "../../../../bin")
# 查询到AlpPython
sys.path.append(py3_path)

# 导入提供的python库
from AlpPython import *
import numpy as np
import cv2 as cv
import threading
import time

def selectEVB(device):
    """
    检查并选择是否使用 EVB 设备。
    该函数获取当前支持的 EVB 设备列表，并提供用户选择一个设备进行配置。
    用户输入设备索引后，程序会根据选择的索引进行设备选择。
    如果设备选择成功，返回 `true`；如果没有设备或选择失败，则返回 `false`。

    :param device: 设备实例

    :return: True 成功 False 失败
    """

    # 1.EVB 设备列表
    device_list = device.getCurrentSupportDevices()

    # 检查设备列表是否为空
    if not device_list:
        print("No devices available!")
        return False

    if len(device_list) == 1:
        if not device.selectCurrentDevice(0):
            print("Failed to select the EVB device!")
            return False

        print(f"You chose {device_list[0].root_config_name} {device_list[0].sensor_name[0]}")
        return True

    # 显示设备列表
    for i, dev_info in enumerate(device_list):
        print(f"{i}. {dev_info.root_config_name} {dev_info.sensor_name[0]}")

    # 获取用户输入并验证
    device_num = -1
    while True:
        try:
            device_num = int(input(f"Please select the required device configuration number [0,{len(device_list) - 1}]: "))
            
            # 输入无效，提示重新输入
            if device_num < 0 or device_num >= len(device_list):
                print(f"The input is invalid. Please enter a valid number between 0 and {len(device_list) - 1}.")
            else:
                break  # 输入有效，退出循环
        except ValueError:
            print("The input is invalid. Please enter a valid number.")

    # 2.  选择设备
    if not device.selectCurrentDevice(device_num):
        print("Failed to select the EVB device!")
        return False

    print(f"You chose {device_list[device_num].root_config_name} {device_list[device_num].sensor_name[0]}")
    return True

def syncFrameCallback(device):
    """
    该函数用于处理从设备接收到同步数据帧

    :param device: 设备实例
    """

    cv.namedWindow("Sync", cv.WINDOW_NORMAL)

    ##七.  监控设备状态##
    while device.isOpened():

        # 5.3 获取同步数据
        sync_list = device.getSyncFrames()

        for it in sync_list:
            aps_image = it[0].convertTo()
            if len(it[1]) == 0:
                continue
            evs_image = it[1][0].frame() # 只取第一个 EVS 帧

            aps_image_resized = cv.resize(aps_image, (evs_image.shape[1], evs_image.shape[0]), interpolation=cv.INTER_LINEAR)
            result = cv.hconcat([aps_image_resized, evs_image * 100])
            cv.imshow("Sync", result)
            cv.waitKey(1)

    cv.destroyWindow("Sync")


def getSensorInfo(device):
    """
    该函数用于获取并打印设备的各种传感器信息，包括配置版本、固件版本、FPGA 版本、曝光时间、帧率、增益等

    :param device: 设备实例
    """

    # 1.配置版本
    print(f"config version: {device.getConfigVersion()}")
    # 2.固件版本
    print(f"firmware version: {device.getFirmwareVersion()}")
    # 3.FPGA版本
    print(f"fpga version: {device.getFpgaVersion()}")

    # APS 参数
    if device.apsModeIndex() > -1:
        # 1.曝光时间(us)
        print(f"aps exposure time: {device.apsExposureTime()} us")
        # 2.帧率
        print(f"aps fps: {device.apsFps()}")
        # 3.模拟增益
        print(f"aps analog gain: {device.apsAnalogGain()}")
        # 4.数据模式
        print(f"aps mode: {device.apsModeString()}")
        # 5.宽高
        print(f"aps width: {device.apsWidth()}")
        print(f"aps height: {device.apsHeight()}")

    # EVS 参数
    if device.evsModeIndex() > -1:
        # 1.帧率
        print(f"evs fps: {device.evsFps()}")
        # 2.灵敏度
        print(f"evs sensitivity: {device.evsSensitivity()}")
        # 3.数据模式
        print(f"evs mode: {device.evsModeString()}")
        # 4.宽高
        print(f"evs width: {device.evsWidth()}")
        print(f"evs height: {device.evsHeight()}")


def setCameraParameters(device):
    """
    该函数用于初始化设备的各种参数，模拟增益、帧率、曝光时间、灵敏度等。

    :param device: 设备实例

    :return: true 表示初始化成功，false 表示初始化失败。
    """

    # APS 参数设置
    if device.apsModeIndex() > -1:

        # 1.  设置 APS 的模拟增益
        rel = device.setApsAnalogGain(1)
        if rel == -1:
            print("Lower machine error - Failed to set analog gain!")
            return False

        # 2.  设置 APS 的帧率
        if not device.setApsFps(15):
            print("Lower machine error - Failed to set APS FPS!")
            return False

        # 3.  设置 APS 的曝光时间(单位 us)
        rel = device.setApsExposureTime(25 * 1000)
        if rel == -1:
            print("Lower machine error - Failed to set APS exposure time!")
            return False


    # EVS 参数设置
    if device.evsModeIndex() > -1:

        # 1.  设置 EVS 的帧率
        if not device.setEvsFps(500):
            print("Lower machine error - Failed to set EVS FPS!")
            return False

        # 2.  设置 EVS 的灵敏度
        if not device.setEvsSensitivity(4):
            print("Lower machine error - Failed to set EVS sensitivity!")
            return False

    return True


# SDK 基本调用流程示例，同步数据流
def main():

    ##一.  设备实例化##
    device = EigerDevice003CA()

    aps_mode = "NORMAL_V2"  # 存储 APS 数据模式
    evs_mode = "NORMAL_V2"  # 存储 EVS 数据模式
    cfg = os.path.join(current_dir, "../../../../config/EVB/APX003CE_COB/003ce_hvs_master_bitformat_972fps_v6.0_new.data")  # 存储配置文件路径

    ##二.  初始化设备##
    recode = device.init(aps_mode, evs_mode, DeviceLinkType.MIDDLE)

    if recode != ErrorCode.NONE:
        # 设备初始化失败，输出错误信息并退出
        errorMsg(recode)
        return

    ##三.  选择并配置 EVB 设备##
    if not selectEVB(device):
        return

    ##四.  打开设备##
    recode = device.open(cfg)

    if recode != ErrorCode.NONE:
        errorMsg(recode)
        return

    # 获取sensor信息
    print(" ------------- get Default configuration -------------")
    getSensorInfo(device)
    print(" ------------- Default configuration end -------------")

    # 打开设备后在下发修改的sensor配置
    if not setCameraParameters(device):
        return

    print(" ------------- get init configuration -------------")
    getSensorInfo(device)
    print(" ------------- init configuration end -------------")

    ##五.  添加数据回调##
    # 5.1 开启回调数据流
    device.startStream()

    # 5.2 开启数据同步
    device.enableSync(True)

    ##六.  启动数据流##
    if not device.start():
        print("Start the data flow failure!")
        return

    t1 = threading.Thread(target = syncFrameCallback, args = (device,))

    t1.start()

    ##七.  监控设备状态##
    while device.isOpened():
        user_input = input("Enter 'q' to end the program: ")
        if user_input.lower() == 'q':
            # 5.4 停止回调数据流
            device.stopStream()
            ##八.  停止数据流##
            device.stop()
            ##九.  关闭设备##
            device.close()

    t1.join()

    
if __name__ == "__main__":
    main()
