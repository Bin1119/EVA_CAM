#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for EVA_CAM system
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List


def setup_paths():
    """Setup system paths for local AlpLib and xArm SDK"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add local AlpLib bin directory to path
    alplib_bin = os.path.join(current_dir, 'alplib', 'bin')
    if os.path.exists(alplib_bin):
        sys.path.append(alplib_bin)
    
    # Add local xArm SDK to path
    xarm_path = os.path.join(current_dir, 'xarm')
    if os.path.exists(xarm_path):
        sys.path.append(xarm_path)
    
    return current_dir


def get_timestamp() -> str:
    """Get current timestamp as string"""
    now = datetime.now()
    milliseconds = now.microsecond // 1000
    formatted_time = now.strftime('%Y%m%d_%H%M%S')
    milliseconds_str = f"{milliseconds:03d}"
    return f"{formatted_time}_{milliseconds_str}"


def create_output_directory(base_dir: str, prefix: str = "experiment") -> str:
    """
    Create output directory with timestamp
    
    Args:
        base_dir: Base directory for output
        prefix: Prefix for directory name
        
    Returns:
        Path to created directory
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    timestamp = get_timestamp()
    output_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def safe_sleep(duration: float):
    """Safe sleep function that can be interrupted"""
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.01)


class DataSynchronizer:
    """Synchronize data collection with robot motion"""
    
    def __init__(self):
        self.is_collecting = False
        self.collection_lock = threading.Lock()
        self.start_time = None
        self.end_time = None
    
    def start_collection(self):
        """Start data collection"""
        with self.collection_lock:
            self.is_collecting = True
            self.start_time = time.time()
    
    def stop_collection(self):
        """Stop data collection"""
        with self.collection_lock:
            self.is_collecting = False
            self.end_time = time.time()
    
    def is_active(self) -> bool:
        """Check if collection is active"""
        with self.collection_lock:
            return self.is_collecting
    
    def get_duration(self) -> Optional[float]:
        """Get collection duration"""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class MotionValidator:
    """Validate robot motion parameters"""
    
    @staticmethod
    def validate_position(x: float, y: float, z: float, 
                         roll: float = None, pitch: float = None, yaw: float = None) -> bool:
        """
        Validate position parameters
        
        Args:
            x, y, z: Position coordinates
            roll, pitch, yaw: Orientation angles (optional)
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - adjust according to robot workspace
        if not (-1000 <= x <= 1000):
            return False
        if not (-1000 <= y <= 1000):
            return False
        if not (-1000 <= z <= 1000):
            return False
        
        # Validate angles if provided
        for angle in [roll, pitch, yaw]:
            if angle is not None and not (-360 <= angle <= 360):
                return False
        
        return True
    
    @staticmethod
    def validate_joint_angles(angles: List[float]) -> bool:
        """
        Validate joint angle parameters
        
        Args:
            angles: List of joint angles
            
        Returns:
            True if valid, False otherwise
        """
        if len(angles) != 7:  # xArm7 has 7 joints
            return False
        
        for angle in angles:
            if not (-360 <= angle <= 360):
                return False
        
        return True
    
    @staticmethod
    def validate_speed(speed: float) -> bool:
        """
        Validate speed parameter
        
        Args:
            speed: Speed value
            
        Returns:
            True if valid, False otherwise
        """
        return 0 < speed <= 1000


class EmergencyHandler:
    """Handle emergency situations"""
    
    def __init__(self):
        self.emergency_stop = False
        self.emergency_lock = threading.Lock()
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        with self.emergency_lock:
            self.emergency_stop = True
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        with self.emergency_lock:
            self.emergency_stop = False
    
    def is_emergency_active(self) -> bool:
        """Check if emergency stop is active"""
        with self.emergency_lock:
            return self.emergency_stop


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default