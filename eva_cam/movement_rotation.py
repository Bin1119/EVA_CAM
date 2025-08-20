#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation movement example - Rotate robot 30 degrees
"""

import sys
import os
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_cam_controller import EvaCamController
from utils import logger


def rotation_yaw_example(no_home=False):
    """Example: Rotate robot 30 degrees around yaw axis while collecting HVS data
    
    Args:
        no_home: If True, skip going to home position during initialization
    """
    
    logger.info("=== Yaw Rotation Example ===")
    
    # Initialize controller
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm(go_home=not no_home):
            logger.error("Failed to initialize xArm")
            return False
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define rotation (30 degrees around yaw axis)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2]
        target_roll = current_pos[3]
        target_pitch = current_pos[4]
        target_yaw = current_pos[5] + 30  # +30 degrees
        
        logger.info(f"Rotating to: yaw={target_yaw}°")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=50, wait=True
        )
        
        if success:
            logger.info("Yaw rotation completed successfully")
            
            # Return to original orientation
            logger.info("Returning to original orientation")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=50, wait=True
            )
        else:
            logger.error("Yaw rotation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in yaw rotation example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def rotation_bidirectional_example(no_home=False):
    """Example: Rotate robot 30 degrees clockwise and counter-clockwise
    
    Args:
        no_home: If True, skip going to home position during initialization
    """
    
    logger.info("=== Bidirectional Rotation Example ===")
    
    # Initialize controller
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm(go_home=not no_home):
            logger.error("Failed to initialize xArm")
            return False
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        rotation_angle = 30  # degrees
        
        # Rotate clockwise 30 degrees
        target_clockwise = [
            current_pos[0],
            current_pos[1],
            current_pos[2],
            current_pos[3],
            current_pos[4],
            current_pos[5] + rotation_angle
        ]
        
        logger.info(f"Rotating clockwise {rotation_angle}°")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_clockwise[0], y=target_clockwise[1], z=target_clockwise[2],
            roll=target_clockwise[3], pitch=target_clockwise[4], yaw=target_clockwise[5],
            speed=40, wait=True
        )
        
        if not success:
            logger.error("Clockwise rotation failed")
            return False
        
        # Pause for 1 second
        import time
        time.sleep(1)
        
        # Rotate counter-clockwise 30 degrees (back to original)
        logger.info(f"Rotating counter-clockwise {rotation_angle}°")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=current_pos[0], y=current_pos[1], z=current_pos[2],
            roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
            speed=40, wait=True
        )
        
        if success:
            logger.info("Bidirectional rotation completed successfully")
        else:
            logger.error("Counter-clockwise rotation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in bidirectional rotation example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def rotation_roll_example(no_home=False):
    """Example: Rotate robot 30 degrees around roll axis
    
    Args:
        no_home: If True, skip going to home position during initialization
    """
    
    logger.info("=== Roll Rotation Example ===")
    
    # Initialize controller
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm(go_home=not no_home):
            logger.error("Failed to initialize xArm")
            return False
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define roll rotation (30 degrees)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2]
        target_roll = current_pos[3] + 30  # +30 degrees
        target_pitch = current_pos[4]
        target_yaw = current_pos[5]
        
        logger.info(f"Rotating to: roll={target_roll}°")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=40, wait=True
        )
        
        if success:
            logger.info("Roll rotation completed successfully")
            
            # Return to original orientation
            logger.info("Returning to original orientation")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=40, wait=True
            )
        else:
            logger.error("Roll rotation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in roll rotation example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def rotation_pitch_example(no_home=False):
    """Example: Rotate robot 30 degrees around pitch axis
    
    Args:
        no_home: If True, skip going to home position during initialization
    """
    
    logger.info("=== Pitch Rotation Example ===")
    
    # Initialize controller
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm(go_home=not no_home):
            logger.error("Failed to initialize xArm")
            return False
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define pitch rotation (30 degrees)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2]
        target_roll = current_pos[3]
        target_pitch = current_pos[4] + 30  # +30 degrees
        target_yaw = current_pos[5]
        
        logger.info(f"Rotating to: pitch={target_pitch}°")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=40, wait=True
        )
        
        if success:
            logger.info("Pitch rotation completed successfully")
            
            # Return to original orientation
            logger.info("Returning to original orientation")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=40, wait=True
            )
        else:
            logger.error("Pitch rotation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in pitch rotation example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def rotation_combined_example(no_home=False):
    """Example: Combined rotation around multiple axes
    
    Args:
        no_home: If True, skip going to home position during initialization
    """
    
    logger.info("=== Combined Rotation Example ===")
    
    # Initialize controller
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm(go_home=not no_home):
            logger.error("Failed to initialize xArm")
            return False
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define combined rotation (15 degrees each axis)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2]
        target_roll = current_pos[3] + 15  # +15 degrees
        target_pitch = current_pos[4] + 15  # +15 degrees
        target_yaw = current_pos[5] + 15  # +15 degrees
        
        logger.info(f"Rotating to: roll={target_roll}°, pitch={target_pitch}°, yaw={target_yaw}°")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=30, wait=True
        )
        
        if success:
            logger.info("Combined rotation completed successfully")
            
            # Return to original orientation
            logger.info("Returning to original orientation")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=30, wait=True
            )
        else:
            logger.error("Combined rotation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in combined rotation example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


if __name__ == "__main__":
    print("Rotation Movement Examples")
    print("1. Yaw rotation (30°)")
    print("2. Bidirectional rotation")
    print("3. Roll rotation (30°)")
    print("4. Pitch rotation (30°)")
    print("5. Combined rotation (15° each axis)")
    
    choice = input("Select example (1-5): ").strip()
    
    if choice == "1":
        rotation_yaw_example()
    elif choice == "2":
        rotation_bidirectional_example()
    elif choice == "3":
        rotation_roll_example()
    elif choice == "4":
        rotation_pitch_example()
    elif choice == "5":
        rotation_combined_example()
    else:
        print("Invalid choice")
        rotation_yaw_example()  # Default to first example