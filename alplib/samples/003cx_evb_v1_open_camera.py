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
 # ##五.  启动数据流##：
 #    - 使用 `start()` 方法启动设备的数据流，开始采集数据。
 #    - 此时，设备进入实时数据采集模式，可以开始获取 APS 或 EVS 数据。
 #
 # ##六.  监控设备状态##：
 #    - 调用 `isOpened()` 方法。
 #    - 持续检查设备的运行状态，确保设备能够正常运行并且数据流稳定。
 #    - 可以调用设备提供的状态检查方法，查看设备是否有异常或错误。
 #
 # ##七.  停止数据流##：
 #    - 在结束操作时，调用 `stop()` 停止设备的数据流。
 #
 # ##八.  关闭设备##：
 #    - 使用 `close()` 方法关闭设备，释放设备占用的资源。
 #
 # 以上步骤是设备使用的完整流程，按照此流程操作能够确保设备正确运行和关闭。
 # 设备从初始化、配置到数据流管理、状态监控，再到停止和资源释放，所有环节都应完整执行。
##

import os
import sys

# 设置路径为本地alplib目录
current_dir = os.path.dirname(os.path.abspath(__file__))
py3_path = os.path.join(current_dir, "..\\bin")
sys.path.append(py3_path)

# 导入提供的python库
from AlpPython import *
import numpy as np
import cv2 as cv
import threading
import time

def chooseModel():
    """
      选择相机的解码模式
      设备支持选择不同的解码模式，可以选择APS数据模式或者EVS数据模式，或者同时选择两者。
      根据用户输入设置适合的模式和配置文件。

      :param aps_mode: APS 数据解码模式，可选值:
        - "NONE" 或 "": 表示不处理 APS 数据
        - "NORMAL_V2": 从设备中获得 APS 数据

      :param evs_mode: EVS EVS 模式，控制 EVS 数据解码方式
        - "NONE" 或 "": 表示不处理 EVS 数据
        - "NORMAL_V2": Normal 2bit 数据
        - "EVENT_V2": Event 16 pixel 数据

      :param cfg: 下发的配置

      :return: 包含三个元素的元组 (aps_mode, evs_mode, cfg)，分别表示 APS 数据解码模式、EVS 数据解码模式和配置文件路径。

    """
    print("Please choose an action:")
    print("Enter 'APS' to output APS data")
    print("Enter 'EVS' to output EVS data")
    print("Enter 'HVS' or any key to output both APS and EVS data")
    strcmd = input("Please enter the command: ").strip().lower()
    
    if strcmd == "aps":
        return "NORMAL_V2", "", os.path.join(current_dir, "..\\config\\APX003CE_COB\\003ce_hvs_master_bitformat_972fps_v6.0_new.data")
    elif strcmd == "evs":
        return "", "NORMAL_V2", os.path.join(current_dir, "..\\config\\APX003CE_COB\\003ce_hvs_master_bitformat_972fps_v6.0_new.data")
    else:
        return "NORMAL_V2", "NORMAL_V2", os.path.join(current_dir, "..\\config\\APX003CE_COB\\003ce_hvs_master_bitformat_972fps_v6.0_new.data")


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


# SDK 基本调用流程示例,打开设备
def main():

    ##一.  设备实例化##
    device = EigerDevice003CA()

    aps_mode = "NORMAL_V2"  # 存储 APS 数据模式
    evs_mode = "NORMAL_V2"  # 存储 EVS 数据模式
    cfg = os.path.join(current_dir, "..\\config\\APX003CE_COB\\003ce_hvs_master_bitformat_972fps_v6.0_new.data")  # 存储配置文件路径

    # 如果使用默认值就可以不调用这个函数
    # 调用 chooseModel 函数选择解码模式，根据用户输入配置 APS 模式和 EVS 模式。
    #aps_mode, evs_mode, cfg = chooseModel()

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

    ##五.  启动数据流##
    if not device.start():
        print("Start the data flow failure!")
        return

    count = 0
    ##六.  监控设备状态##
    while device.isOpened():
        print("device is running!")
        count += 1
        if count > 10:
            ##七.  停止数据流##
            device.stop()
            ##八.  关闭设备##
            device.close()
        time.sleep(1)  
    
if __name__ == "__main__":
    main()
