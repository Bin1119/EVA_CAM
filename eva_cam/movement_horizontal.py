#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizontal movement example - Move robot 10cm horizontally
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_cam_controller import EvaCamController
from utils import logger


def horizontal_movement_example():
    """Example: Move robot 10cm horizontally while collecting HVS data"""
    
    logger.info("=== Horizontal Movement Example ===")
    
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
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define horizontal movement (10cm in X direction)
        target_x = current_pos[0] + 100  # +100mm = +10cm
        target_y = current_pos[1]
        target_z = current_pos[2]
        target_roll = current_pos[3]
        target_pitch = current_pos[4]
        target_yaw = current_pos[5]
        
        logger.info(f"Moving to: x={target_x}, y={target_y}, z={target_z}")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=100, wait=True
        )
        
        if success:
            logger.info("Horizontal movement completed successfully")
            
            # Return to original position
            logger.info("Returning to original position")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=100, wait=True
            )
        else:
            logger.error("Horizontal movement failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in horizontal movement example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def horizontal_movement_bidirectional_example():
    """Example: Move robot 10cm forward and backward while collecting HVS data"""
    
    logger.info("=== Bidirectional Horizontal Movement Example ===")
    
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
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define movement parameters
        movement_distance = 100  # 100mm = 10cm
        
        # Move forward 10cm
        target_forward = [
            current_pos[0] + movement_distance,
            current_pos[1],
            current_pos[2],
            current_pos[3],
            current_pos[4],
            current_pos[5]
        ]
        
        logger.info("Moving forward 10cm")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_forward[0], y=target_forward[1], z=target_forward[2],
            roll=target_forward[3], pitch=target_forward[4], yaw=target_forward[5],
            speed=80, wait=True
        )
        
        if not success:
            logger.error("Forward movement failed")
            return False
        
        # Pause for 1 second
        import time
        time.sleep(1)
        
        # Move backward 10cm (back to original position)
        logger.info("Moving backward 10cm")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=current_pos[0], y=current_pos[1], z=current_pos[2],
            roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
            speed=80, wait=True
        )
        
        if success:
            logger.info("Bidirectional horizontal movement completed successfully")
        else:
            logger.error("Backward movement failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in bidirectional horizontal movement example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def horizontal_movement_y_direction_example():
    """Example: Move robot 10cm in Y direction while collecting HVS data"""
    
    logger.info("=== Y Direction Horizontal Movement Example ===")
    
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
        
        # Get current position
        current_pos = controller.arm.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Define Y direction movement (10cm in Y direction)
        target_x = current_pos[0]
        target_y = current_pos[1] + 100  # +100mm = +10cm
        target_z = current_pos[2]
        target_roll = current_pos[3]
        target_pitch = current_pos[4]
        target_yaw = current_pos[5]
        
        logger.info(f"Moving to: x={target_x}, y={target_y}, z={target_z}")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=100, wait=True
        )
        
        if success:
            logger.info("Y direction horizontal movement completed successfully")
            
            # Return to original position
            logger.info("Returning to original position")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=100, wait=True
            )
        else:
            logger.error("Y direction horizontal movement failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in Y direction horizontal movement example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


if __name__ == "__main__":
    print("Horizontal Movement Examples")
    print("1. Simple horizontal movement (X direction)")
    print("2. Bidirectional horizontal movement")
    print("3. Y direction horizontal movement")
    
    choice = input("Select example (1-3): ").strip()
    
    if choice == "1":
        horizontal_movement_example()
    elif choice == "2":
        horizontal_movement_bidirectional_example()
    elif choice == "3":
        horizontal_movement_y_direction_example()
    else:
        print("Invalid choice")
        horizontal_movement_example()  # Default to first example