#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main controller class for EVA_CAM system
Integrates xArm robot control with AlpLib HVS data acquisition
"""

import os
import sys
import time
import threading
import cv2 as cv
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime

# Setup paths and import required modules
from utils import setup_paths, logger, config, DataSynchronizer, MotionValidator, EmergencyHandler
setup_paths()

try:
    from xarm.wrapper import XArmAPI
    from AlpPython import *
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


class EvaCamController:
    """Main controller for EVA_CAM system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize EVA_CAM controller
        
        Args:
            config_path: Path to configuration file
        """
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.arm: Optional[XArmAPI] = None
        self.camera: Optional[EigerDevice003CA] = None
        self.writer_file: Optional[WriterFile] = None
        
        # Synchronization and control
        self.data_sync = DataSynchronizer()
        self.motion_validator = MotionValidator()
        self.emergency_handler = EmergencyHandler()
        
        # Threading
        self.aps_thread: Optional[threading.Thread] = None
        self.evs_thread: Optional[threading.Thread] = None
        self.camera_threads_running = False
        
        # Data collection state
        self.is_collecting = False
        self.current_output_dir = None
        self.session_id = None
        
        # Load configuration
        self.xarm_config = self.config.get_xarm_config()
        self.camera_config = self.config.get_camera_config()
        self.recording_config = self.config.get_recording_config()
        self.motion_config = self.config.get_motion_config()
        self.safety_config = self.config.get_safety_config()
        
        self.logger.info("EVA_CAM Controller initialized")
    
    def initialize_xarm(self) -> bool:
        """
        Initialize xArm robot connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to xArm at {self.xarm_config['ip']}")
            self.arm = XArmAPI(self.xarm_config['ip'])
            
            # Enable motion and set initial state
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)  # Position mode
            self.arm.set_state(state=0)
            
            # Set safety parameters
            if self.safety_config['enable_collision_detection']:
                self.arm.set_collision_sensitivity(self.safety_config['collision_sensitivity'])
            
            # Go to home position
            self.arm.move_gohome(wait=True)
            
            self.logger.info("xArm initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize xArm: {e}")
            return False
    
    def initialize_camera(self) -> bool:
        """
        Initialize AlpLib camera
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing AlpLib camera")
            
            # Get configuration file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(current_dir, 'alplib', 'config', 
                                   'APX003CE_COB', '003ce_hvs_master_bitformat_972fps_v6.0_new.data')
            
            # Create camera instance
            self.camera = EigerDevice003CA()
            
            # Initialize camera
            recode = self.camera.init(
                self.camera_config['aps_mode'],
                self.camera_config['evs_mode'],
                DeviceLinkType.MIDDLE
            )
            
            if recode != ErrorCode.NONE:
                self.logger.error(f"Camera initialization failed: {recode}")
                return False
            
            # Select EVB device
            if not self._select_evb_device():
                return False
            
            # Open camera
            recode = self.camera.open(cfg_path)
            if recode != ErrorCode.NONE:
                self.logger.error(f"Failed to open camera: {recode}")
                return False
            
            # Set camera parameters
            if not self._set_camera_parameters():
                return False
            
            # Start data stream
            self.camera.startStream()
            if not self.camera.start():
                self.logger.error("Failed to start camera data stream")
                return False
            
            # Initialize writer file
            self.writer_file = WriterFile()
            
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _select_evb_device(self) -> bool:
        """Select EVB device"""
        try:
            device_list = self.camera.getCurrentSupportDevices()
            
            if not device_list:
                self.logger.error("No devices available")
                return False
            
            # Select first device if only one available
            if len(device_list) == 1:
                if not self.camera.selectCurrentDevice(0):
                    self.logger.error("Failed to select EVB device")
                    return False
                
                self.logger.info(f"Selected device: {device_list[0].root_config_name} {device_list[0].sensor_name[0]}")
                return True
            
            # For multiple devices, select the first one (can be customized)
            if not self.camera.selectCurrentDevice(0):
                self.logger.error("Failed to select EVB device")
                return False
            
            self.logger.info(f"Selected device: {device_list[0].root_config_name} {device_list[0].sensor_name[0]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to select EVB device: {e}")
            return False
    
    def _set_camera_parameters(self) -> bool:
        """Set camera parameters"""
        try:
            # Set APS parameters
            if self.camera.apsModeIndex() > -1:
                # Set analog gain
                rel = self.camera.setApsAnalogGain(self.camera_config['aps_analog_gain'])
                if rel == -1:
                    self.logger.error("Failed to set APS analog gain")
                    return False
                
                # Set FPS
                if not self.camera.setApsFps(self.camera_config['aps_fps']):
                    self.logger.error("Failed to set APS FPS")
                    return False
                
                # Set exposure time
                rel = self.camera.setApsExposureTime(self.camera_config['aps_exposure_time'])
                if rel == -1:
                    self.logger.error("Failed to set APS exposure time")
                    return False
            
            # Set EVS parameters
            if self.camera.evsModeIndex() > -1:
                # Set FPS
                if not self.camera.setEvsFps(self.camera_config['evs_fps']):
                    self.logger.error("Failed to set EVS FPS")
                    return False
                
                # Set sensitivity
                if not self.camera.setEvsSensitivity(self.camera_config['evs_sensitivity']):
                    self.logger.error("Failed to set EVS sensitivity")
                    return False
            
            self.logger.info("Camera parameters set successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set camera parameters: {e}")
            return False
    
    def start_data_collection(self) -> bool:
        """
        Start synchronized data collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_collecting:
                self.logger.warning("Data collection already in progress")
                return False
            
            # Create output directory
            self.session_id = self._get_timestamp()
            self.current_output_dir = os.path.join(
                self.recording_config['output_dir'],
                f"session_{self.session_id}"
            )
            os.makedirs(self.current_output_dir, exist_ok=True)
            
            # Initialize recording
            if not self._init_recording():
                return False
            
            # Start data collection
            self.data_sync.start_collection()
            self.is_collecting = True
            
            # Start camera data threads
            self._start_camera_threads()
            
            self.logger.info(f"Data collection started - Session ID: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start data collection: {e}")
            return False
    
    def stop_data_collection(self) -> bool:
        """
        Stop data collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_collecting:
                self.logger.warning("Data collection not in progress")
                return False
            
            # Stop data collection
            self.is_collecting = False
            self.data_sync.stop_collection()
            
            # Stop camera threads
            self._stop_camera_threads()
            
            # Close recording file
            if self.writer_file:
                self.writer_file.close()
            
            duration = self.data_sync.get_duration()
            self.logger.info(f"Data collection stopped - Duration: {duration:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop data collection: {e}")
            return False
    
    def _init_recording(self) -> bool:
        """Initialize recording"""
        try:
            if not self.writer_file:
                self.logger.error("Writer file not initialized")
                return False
            
            # Determine file format and extension
            if self.recording_config['format'] == 'bin':
                file_name = os.path.join(self.current_output_dir, "data")
                writer_type = WriterType.BIN
            else:
                file_name = os.path.join(self.current_output_dir, "data.alpdata")
                writer_type = WriterType.HDF5
            
            # Get device attributes
            device_attribute = self.camera.getDeviceAttribute(writer_type, file_name)
            aps_ptr = None
            evs_ptr = None
            
            if self.camera.apsModeIndex() > -1:
                aps_ptr = self.camera.getApsAttribute()
            
            if self.camera.evsModeIndex() > -1:
                evs_ptr = self.camera.getEvsAttribute()
            
            # Open file
            error = self.writer_file.open(device_attribute, aps_ptr, evs_ptr)
            if error != AlpSaveFileError.NONE:
                self.logger.error(f"Failed to open recording file: {error}")
                return False
            
            self.logger.info("Recording initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize recording: {e}")
            return False
    
    def _start_camera_threads(self):
        """Start camera data collection threads"""
        if not self.camera_threads_running:
            self.camera_threads_running = True
            
            if self.camera.apsModeIndex() > -1:
                self.aps_thread = threading.Thread(
                    target=self._aps_data_thread,
                    daemon=True
                )
                self.aps_thread.start()
            
            if self.camera.evsModeIndex() > -1:
                self.evs_thread = threading.Thread(
                    target=self._evs_data_thread,
                    daemon=True
                )
                self.evs_thread.start()
    
    def _stop_camera_threads(self):
        """Stop camera data collection threads"""
        self.camera_threads_running = False
        
        if self.aps_thread and self.aps_thread.is_alive():
            self.aps_thread.join(timeout=1.0)
        
        if self.evs_thread and self.evs_thread.is_alive():
            self.evs_thread.join(timeout=1.0)
    
    def _aps_data_thread(self):
        """APS data collection thread"""
        try:
            cv.namedWindow("APS", cv.WINDOW_NORMAL)
            
            while self.camera_threads_running and self.camera.isOpened():
                frames = self.camera.getApsFrames()
                
                # Save data if collecting
                if self.is_collecting and frames:
                    for frame in frames:
                        self.writer_file.write(frame)
                
                # Display frames
                for frame in frames:
                    aps_image = frame.convertTo()
                    cv.imshow("APS", aps_image)
                    cv.waitKey(1)
                    break
            
            cv.destroyWindow("APS")
            
        except Exception as e:
            self.logger.error(f"APS data thread error: {e}")
    
    def _evs_data_thread(self):
        """EVS data collection thread"""
        try:
            cv.namedWindow("EVS", cv.WINDOW_NORMAL)
            
            while self.camera_threads_running and self.camera.isOpened():
                frames = self.camera.getEvsFrames()
                
                # Save data if collecting
                if self.is_collecting and frames:
                    for frame in frames:
                        self.writer_file.write(frame)
                
                # Display frames
                for frame in frames:
                    evs_image = frame.frame()
                    cv.imshow("EVS", evs_image * 100)
                    cv.waitKey(1)
                    break
            
            cv.destroyWindow("EVS")
            
        except Exception as e:
            self.logger.error(f"EVS data thread error: {e}")
    
    def execute_motion_with_data_collection(self, motion_function: Callable, *args, **kwargs) -> bool:
        """
        Execute motion with synchronized data collection
        
        Args:
            motion_function: Function that performs the motion
            *args, **kwargs: Arguments for the motion function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.emergency_handler.is_emergency_active():
                self.logger.error("Emergency stop active - cannot execute motion")
                return False
            
            self.logger.info("Starting motion with data collection")
            
            # Start data collection
            if not self.start_data_collection():
                self.logger.error("Failed to start data collection")
                return False
            
            # Execute motion
            try:
                result = motion_function(*args, **kwargs)
                if not result:
                    self.logger.error("Motion execution failed")
                    return False
            except Exception as e:
                self.logger.error(f"Motion execution error: {e}")
                return False
            finally:
                # Stop data collection
                self.stop_data_collection()
            
            self.logger.info("Motion and data collection completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in execute_motion_with_data_collection: {e}")
            return False
    
    def move_linear(self, x: float, y: float, z: float, 
                   roll: float = None, pitch: float = None, yaw: float = None,
                   speed: float = None, wait: bool = True) -> bool:
        """
        Move robot in linear motion
        
        Args:
            x, y, z: Target position
            roll, pitch, yaw: Target orientation (optional)
            speed: Movement speed
            wait: Wait for completion
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.arm:
                self.logger.error("xArm not initialized")
                return False
            
            # Validate parameters
            if not self.motion_validator.validate_position(x, y, z, roll, pitch, yaw):
                self.logger.error("Invalid position parameters")
                return False
            
            speed = speed or self.motion_config['linear_speed']
            
            # Execute movement
            result = self.arm.set_position(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                speed=speed, wait=wait
            )
            
            if result == 0:
                self.logger.info(f"Linear motion completed: x={x}, y={y}, z={z}")
                return True
            else:
                self.logger.error(f"Linear motion failed: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Linear motion error: {e}")
            return False
    
    def move_joint(self, angles: List[float], speed: float = None, wait: bool = True) -> bool:
        """
        Move robot joints
        
        Args:
            angles: Joint angles
            speed: Movement speed
            wait: Wait for completion
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.arm:
                self.logger.error("xArm not initialized")
                return False
            
            # Validate parameters
            if not self.motion_validator.validate_joint_angles(angles):
                self.logger.error("Invalid joint angles")
                return False
            
            speed = speed or self.motion_config['angular_speed']
            
            # Execute movement
            result = self.arm.set_servo_angle(angle=angles, speed=speed, wait=wait)
            
            if result == 0:
                self.logger.info(f"Joint motion completed: angles={angles}")
                return True
            else:
                self.logger.error(f"Joint motion failed: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Joint motion error: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop"""
        try:
            self.logger.warning("EMERGENCY STOP ACTIVATED")
            self.emergency_handler.trigger_emergency_stop()
            
            if self.arm:
                self.arm.emergency_stop()
            
            # Stop data collection
            self.stop_data_collection()
            
            self.logger.info("Emergency stop completed")
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    def reset_system(self):
        """Reset system after emergency stop"""
        try:
            self.logger.info("Resetting system")
            self.emergency_handler.reset_emergency_stop()
            
            if self.arm:
                self.arm.motion_enable(enable=True)
                self.arm.set_mode(0)
                self.arm.set_state(state=0)
                self.arm.clean_error()
                self.arm.clean_warn()
            
            self.logger.info("System reset completed")
            
        except Exception as e:
            self.logger.error(f"System reset error: {e}")
    
    def disconnect(self):
        """Disconnect all systems"""
        try:
            self.logger.info("Disconnecting EVA_CAM system")
            
            # Stop data collection
            self.stop_data_collection()
            
            # Disconnect xArm
            if self.arm:
                self.arm.disconnect()
            
            # Close camera
            if self.camera:
                self.camera.stop()
                self.camera.stopStream()
                self.camera.close()
            
            self.logger.info("EVA_CAM system disconnected")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
    def _get_timestamp(self) -> str:
        """Get timestamp string"""
        now = datetime.now()
        milliseconds = now.microsecond // 1000
        formatted_time = now.strftime('%Y%m%d_%H%M%S')
        milliseconds_str = f"{milliseconds:03d}"
        return f"{formatted_time}_{milliseconds_str}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'xarm_connected': self.arm is not None,
            'camera_connected': self.camera is not None,
            'is_collecting': self.is_collecting,
            'emergency_active': self.emergency_handler.is_emergency_active(),
            'session_id': self.session_id,
            'output_dir': self.current_output_dir
        }
        
        if self.data_sync.get_duration():
            status['collection_duration'] = self.data_sync.get_duration()
        
        return status