#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utilities for EVA_CAM system
"""

import os
import configparser
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager for EVA_CAM system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', 'config', 'settings.conf')
        
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
        else:
            print(f"Warning: Config file not found at {self.config_path}")
    
    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get configuration value as integer"""
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get configuration value as float"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get configuration value as boolean"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def get_xarm_config(self) -> Dict[str, Any]:
        """Get xArm configuration"""
        return {
            'ip': self.get('XARM', 'ip', fallback='192.168.1.113'),
            'port': self.getint('XARM', 'port', fallback=3000),
            'timeout': self.getint('XARM', 'timeout', fallback=10),
            'default_speed': self.getint('XARM', 'default_speed', fallback=100),
            'default_acc': self.getint('XARM', 'default_acc', fallback=2000)
        }
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return {
            'mode': self.get('CAMERA', 'mode', fallback='HVS'),
            'aps_mode': self.get('CAMERA', 'aps_mode', fallback='NORMAL_V2'),
            'evs_mode': self.get('CAMERA', 'evs_mode', fallback='NORMAL_V2'),
            'aps_exposure_time': self.getint('APS', 'exposure_time_us', fallback=25000),
            'aps_fps': self.getint('APS', 'fps', fallback=15),
            'aps_analog_gain': self.getint('APS', 'analog_gain', fallback=1),
            'evs_fps': self.getint('EVS', 'fps', fallback=500),
            'evs_sensitivity': self.getint('EVS', 'sensitivity', fallback=4)
        }
    
    def get_recording_config(self) -> Dict[str, Any]:
        """Get recording configuration"""
        return {
            'format': self.get('RECORDING', 'format', fallback='hdf5'),
            'output_dir': self.get('RECORDING', 'output_dir', fallback='./data'),
            'create_timestamp_dirs': self.getboolean('RECORDING', 'create_timestamp_dirs', fallback=True)
        }
    
    def get_motion_config(self) -> Dict[str, Any]:
        """Get motion configuration"""
        return {
            'linear_speed': self.getint('MOTION', 'linear_speed', fallback=100),
            'angular_speed': self.getint('MOTION', 'angular_speed', fallback=50),
            'rotation_speed': self.getint('MOTION', 'rotation_speed', fallback=30),
            'vertical_speed': self.getint('MOTION', 'vertical_speed', fallback=100)
        }
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety configuration"""
        return {
            'collision_sensitivity': self.getint('SAFETY', 'collision_sensitivity', fallback=5),
            'enable_collision_detection': self.getboolean('SAFETY', 'enable_collision_detection', fallback=True),
            'emergency_stop_enabled': self.getboolean('SAFETY', 'emergency_stop_enabled', fallback=True)
        }


# Global configuration instance
config = ConfigManager()