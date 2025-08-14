#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom motion template for creating personalized movement patterns
This template provides a framework for users to define their own motion sequences
"""

import sys
import os
import time
import math
from typing import List, Dict, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_cam_controller import EvaCamController
from utils import logger, MotionValidator


class MotionPattern(ABC):
    """Abstract base class for motion patterns"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.validator = MotionValidator()
    
    @abstractmethod
    def execute(self, controller: EvaCamController) -> bool:
        """Execute the motion pattern"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get pattern parameters"""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate pattern parameters"""
        return True


class LinearMotionPattern(MotionPattern):
    """Linear motion pattern template"""
    
    def __init__(self):
        super().__init__("Linear Motion", "Move in a straight line")
        self.default_params = {
            'distance_x': 100,  # mm
            'distance_y': 0,     # mm
            'distance_z': 0,     # mm
            'speed': 100,        # mm/s
            'return_to_start': True,
            'pause_at_end': 1.0  # seconds
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.default_params.copy()
    
    def execute(self, controller: EvaCamController, params: Dict[str, Any] = None) -> bool:
        """Execute linear motion"""
        try:
            # Use default parameters if none provided
            if params is None:
                params = self.default_params
            
            # Get current position
            current_pos = controller.arm.get_position()
            logger.info(f"Current position: {current_pos}")
            
            # Calculate target position
            target_x = current_pos[0] + params['distance_x']
            target_y = current_pos[1] + params['distance_y']
            target_z = current_pos[2] + params['distance_z']
            
            # Validate target position
            if not self.validator.validate_position(target_x, target_y, target_z):
                logger.error("Invalid target position")
                return False
            
            logger.info(f"Moving to: x={target_x}, y={target_y}, z={target_z}")
            
            # Execute motion with data collection
            success = controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=target_x, y=target_y, z=target_z,
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=params['speed'], wait=True
            )
            
            if not success:
                logger.error("Linear motion failed")
                return False
            
            # Pause at end position
            if params['pause_at_end'] > 0:
                time.sleep(params['pause_at_end'])
            
            # Return to start position
            if params['return_to_start']:
                logger.info("Returning to start position")
                success = controller.execute_motion_with_data_collection(
                    controller.move_linear,
                    x=current_pos[0], y=current_pos[1], z=current_pos[2],
                    roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                    speed=params['speed'], wait=True
                )
                
                if not success:
                    logger.error("Return to start failed")
                    return False
            
            logger.info("Linear motion pattern completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in linear motion pattern: {e}")
            return False


class RotationalMotionPattern(MotionPattern):
    """Rotational motion pattern template"""
    
    def __init__(self):
        super().__init__("Rotational Motion", "Rotate around specified axes")
        self.default_params = {
            'rotation_roll': 0,    # degrees
            'rotation_pitch': 0,   # degrees
            'rotation_yaw': 30,    # degrees
            'speed': 50,           # degrees/s
            'return_to_start': True,
            'pause_at_end': 1.0    # seconds
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.default_params.copy()
    
    def execute(self, controller: EvaCamController, params: Dict[str, Any] = None) -> bool:
        """Execute rotational motion"""
        try:
            # Use default parameters if none provided
            if params is None:
                params = self.default_params
            
            # Get current position
            current_pos = controller.arm.get_position()
            logger.info(f"Current position: {current_pos}")
            
            # Calculate target orientation
            target_roll = current_pos[3] + params['rotation_roll']
            target_pitch = current_pos[4] + params['rotation_pitch']
            target_yaw = current_pos[5] + params['rotation_yaw']
            
            logger.info(f"Rotating to: roll={target_roll}°, pitch={target_pitch}°, yaw={target_yaw}°")
            
            # Execute motion with data collection
            success = controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=target_roll, pitch=target_pitch, yaw=target_yaw,
                speed=params['speed'], wait=True
            )
            
            if not success:
                logger.error("Rotational motion failed")
                return False
            
            # Pause at end position
            if params['pause_at_end'] > 0:
                time.sleep(params['pause_at_end'])
            
            # Return to start orientation
            if params['return_to_start']:
                logger.info("Returning to start orientation")
                success = controller.execute_motion_with_data_collection(
                    controller.move_linear,
                    x=current_pos[0], y=current_pos[1], z=current_pos[2],
                    roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                    speed=params['speed'], wait=True
                )
                
                if not success:
                    logger.error("Return to start failed")
                    return False
            
            logger.info("Rotational motion pattern completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in rotational motion pattern: {e}")
            return False


class CircularMotionPattern(MotionPattern):
    """Circular motion pattern template"""
    
    def __init__(self):
        super().__init__("Circular Motion", "Move in a circular pattern")
        self.default_params = {
            'radius': 100,         # mm
            'steps': 8,            # number of steps in circle
            'speed': 100,          # mm/s
            'axis': 'z',           # rotation axis ('x', 'y', or 'z')
            'return_to_start': True,
            'pause_at_steps': 0.5  # seconds pause at each step
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.default_params.copy()
    
    def execute(self, controller: EvaCamController, params: Dict[str, Any] = None) -> bool:
        """Execute circular motion"""
        try:
            # Use default parameters if none provided
            if params is None:
                params = self.default_params
            
            # Get current position (center of circle)
            center_pos = controller.arm.get_position()
            logger.info(f"Center position: {center_pos}")
            
            # Generate circular path points
            points = []
            for i in range(params['steps']):
                angle = 2 * math.pi * i / params['steps']
                
                if params['axis'] == 'z':
                    x = center_pos[0] + params['radius'] * math.cos(angle)
                    y = center_pos[1] + params['radius'] * math.sin(angle)
                    z = center_pos[2]
                elif params['axis'] == 'x':
                    x = center_pos[0]
                    y = center_pos[1] + params['radius'] * math.cos(angle)
                    z = center_pos[2] + params['radius'] * math.sin(angle)
                elif params['axis'] == 'y':
                    x = center_pos[0] + params['radius'] * math.cos(angle)
                    y = center_pos[1]
                    z = center_pos[2] + params['radius'] * math.sin(angle)
                else:
                    logger.error(f"Invalid axis: {params['axis']}")
                    return False
                
                points.append([x, y, z, center_pos[3], center_pos[4], center_pos[5]])
            
            # Execute circular motion
            for i, point in enumerate(points):
                logger.info(f"Circular step {i+1}/{params['steps']}: x={point[0]}, y={point[1]}, z={point[2]}")
                
                success = controller.execute_motion_with_data_collection(
                    controller.move_linear,
                    x=point[0], y=point[1], z=point[2],
                    roll=point[3], pitch=point[4], yaw=point[5],
                    speed=params['speed'], wait=True
                )
                
                if not success:
                    logger.error(f"Circular step {i+1} failed")
                    return False
                
                # Pause at each step
                if params['pause_at_steps'] > 0:
                    time.sleep(params['pause_at_steps'])
            
            # Return to start position
            if params['return_to_start']:
                logger.info("Returning to center position")
                success = controller.execute_motion_with_data_collection(
                    controller.move_linear,
                    x=center_pos[0], y=center_pos[1], z=center_pos[2],
                    roll=center_pos[3], pitch=center_pos[4], yaw=center_pos[5],
                    speed=params['speed'], wait=True
                )
                
                if not success:
                    logger.error("Return to center failed")
                    return False
            
            logger.info("Circular motion pattern completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in circular motion pattern: {e}")
            return False


class JointMotionPattern(MotionPattern):
    """Joint motion pattern template"""
    
    def __init__(self):
        super().__init__("Joint Motion", "Move individual joints")
        self.default_params = {
            'joint_angles': [0, 0, 0, 0, 0, 0, 0],  # 7 joints
            'speed': 50,                           # degrees/s
            'return_to_start': True,
            'pause_at_end': 1.0                    # seconds
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.default_params.copy()
    
    def execute(self, controller: EvaCamController, params: Dict[str, Any] = None) -> bool:
        """Execute joint motion"""
        try:
            # Use default parameters if none provided
            if params is None:
                params = self.default_params
            
            # Get current joint angles
            current_angles = controller.arm.get_servo_angle()
            logger.info(f"Current joint angles: {current_angles}")
            
            # Validate joint angles
            if not self.validator.validate_joint_angles(params['joint_angles']):
                logger.error("Invalid joint angles")
                return False
            
            logger.info(f"Moving to joint angles: {params['joint_angles']}")
            
            # Execute motion with data collection
            success = controller.execute_motion_with_data_collection(
                controller.move_joint,
                angles=params['joint_angles'],
                speed=params['speed'], wait=True
            )
            
            if not success:
                logger.error("Joint motion failed")
                return False
            
            # Pause at end position
            if params['pause_at_end'] > 0:
                time.sleep(params['pause_at_end'])
            
            # Return to start position
            if params['return_to_start']:
                logger.info("Returning to start joint angles")
                success = controller.execute_motion_with_data_collection(
                    controller.move_joint,
                    angles=current_angles[:7],  # Only first 7 joints
                    speed=params['speed'], wait=True
                )
                
                if not success:
                    logger.error("Return to start failed")
                    return False
            
            logger.info("Joint motion pattern completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in joint motion pattern: {e}")
            return False


class CustomMotionBuilder:
    """Builder for creating custom motion sequences"""
    
    def __init__(self, controller: EvaCamController):
        self.controller = controller
        self.patterns = {
            'linear': LinearMotionPattern(),
            'rotational': RotationalMotionPattern(),
            'circular': CircularMotionPattern(),
            'joint': JointMotionPattern()
        }
    
    def execute_pattern(self, pattern_name: str, params: Dict[str, Any] = None) -> bool:
        """Execute a motion pattern"""
        if pattern_name not in self.patterns:
            logger.error(f"Unknown pattern: {pattern_name}")
            return False
        
        pattern = self.patterns[pattern_name]
        logger.info(f"Executing pattern: {pattern.name}")
        
        return pattern.execute(self.controller, params)
    
    def execute_sequence(self, sequence: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """Execute a sequence of motion patterns"""
        logger.info(f"Executing motion sequence with {len(sequence)} patterns")
        
        for i, (pattern_name, params) in enumerate(sequence):
            logger.info(f"Sequence step {i+1}: {pattern_name}")
            
            success = self.execute_pattern(pattern_name, params)
            if not success:
                logger.error(f"Sequence step {i+1} failed")
                return False
            
            # Pause between patterns
            time.sleep(0.5)
        
        logger.info("Motion sequence completed successfully")
        return True
    
    def list_patterns(self) -> List[str]:
        """List available patterns"""
        return list(self.patterns.keys())
    
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """Get information about a pattern"""
        if pattern_name not in self.patterns:
            return {}
        
        pattern = self.patterns[pattern_name]
        return {
            'name': pattern.name,
            'description': pattern.description,
            'parameters': pattern.get_parameters()
        }


def create_custom_sequence_example():
    """Example of creating a custom motion sequence"""
    
    logger.info("=== Custom Motion Sequence Example ===")
    
    # Initialize controller
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm():
            logger.error("Failed to initialize xArm")
            return False
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Create motion builder
        builder = CustomMotionBuilder(controller)
        
        # Define custom sequence
        custom_sequence = [
            ('linear', {
                'distance_x': 100,
                'distance_y': 0,
                'distance_z': 0,
                'speed': 100,
                'return_to_start': False,
                'pause_at_end': 0.5
            }),
            ('rotational', {
                'rotation_roll': 0,
                'rotation_pitch': 0,
                'rotation_yaw': 45,
                'speed': 40,
                'return_to_start': False,
                'pause_at_end': 0.5
            }),
            ('linear', {
                'distance_x': 0,
                'distance_y': 100,
                'distance_z': 50,
                'speed': 80,
                'return_to_start': False,
                'pause_at_end': 0.5
            }),
            ('circular', {
                'radius': 50,
                'steps': 6,
                'speed': 60,
                'axis': 'z',
                'return_to_start': False,
                'pause_at_steps': 0.3
            })
        ]
        
        # Execute custom sequence
        success = builder.execute_sequence(custom_sequence)
        
        if success:
            logger.info("Custom motion sequence completed successfully")
        else:
            logger.error("Custom motion sequence failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in custom motion sequence example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


if __name__ == "__main__":
    print("Custom Motion Template")
    print("1. Execute linear motion pattern")
    print("2. Execute rotational motion pattern")
    print("3. Execute circular motion pattern")
    print("4. Execute joint motion pattern")
    print("5. Execute custom sequence example")
    
    choice = input("Select example (1-5): ").strip()
    
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm():
            logger.error("Failed to initialize xArm")
            exit(1)
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            exit(1)
        
        builder = CustomMotionBuilder(controller)
        
        if choice == "1":
            # Linear motion with custom parameters
            params = {
                'distance_x': 150,
                'distance_y': 50,
                'distance_z': 0,
                'speed': 120,
                'return_to_start': True,
                'pause_at_end': 1.0
            }
            builder.execute_pattern('linear', params)
            
        elif choice == "2":
            # Rotational motion with custom parameters
            params = {
                'rotation_roll': 15,
                'rotation_pitch': 30,
                'rotation_yaw': 45,
                'speed': 40,
                'return_to_start': True,
                'pause_at_end': 1.0
            }
            builder.execute_pattern('rotational', params)
            
        elif choice == "3":
            # Circular motion with custom parameters
            params = {
                'radius': 80,
                'steps': 12,
                'speed': 80,
                'axis': 'z',
                'return_to_start': True,
                'pause_at_steps': 0.3
            }
            builder.execute_pattern('circular', params)
            
        elif choice == "4":
            # Joint motion with custom parameters
            params = {
                'joint_angles': [45, -30, 60, 0, -45, 30, 0],
                'speed': 60,
                'return_to_start': True,
                'pause_at_end': 1.0
            }
            builder.execute_pattern('joint', params)
            
        elif choice == "5":
            # Custom sequence example
            create_custom_sequence_example()
            
        else:
            print("Invalid choice")
            # Default linear motion
            builder.execute_pattern('linear')
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    finally:
        controller.disconnect()