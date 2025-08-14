#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertical movement example - Move robot 10cm vertically
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_cam_controller import EvaCamController
from utils import logger


def vertical_movement_up_example():
    """Example: Move robot 10cm up while collecting HVS data"""
    
    logger.info("=== Vertical Movement Up Example ===")
    
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
        
        # Define vertical movement (10cm up in Z direction)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2] + 100  # +100mm = +10cm
        target_roll = current_pos[3]
        target_pitch = current_pos[4]
        target_yaw = current_pos[5]
        
        logger.info(f"Moving up to: z={target_z}")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=80, wait=True
        )
        
        if success:
            logger.info("Vertical movement up completed successfully")
            
            # Return to original position
            logger.info("Returning to original position")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=80, wait=True
            )
        else:
            logger.error("Vertical movement up failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in vertical movement up example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def vertical_movement_down_example():
    """Example: Move robot 10cm down while collecting HVS data"""
    
    logger.info("=== Vertical Movement Down Example ===")
    
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
        
        # Define vertical movement (10cm down in Z direction)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2] - 100  # -100mm = -10cm
        target_roll = current_pos[3]
        target_pitch = current_pos[4]
        target_yaw = current_pos[5]
        
        logger.info(f"Moving down to: z={target_z}")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=80, wait=True
        )
        
        if success:
            logger.info("Vertical movement down completed successfully")
            
            # Return to original position
            logger.info("Returning to original position")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=80, wait=True
            )
        else:
            logger.error("Vertical movement down failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in vertical movement down example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def vertical_movement_bidirectional_example():
    """Example: Move robot 10cm up and down while collecting HVS data"""
    
    logger.info("=== Bidirectional Vertical Movement Example ===")
    
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
        
        movement_distance = 100  # 100mm = 10cm
        
        # Move up 10cm
        target_up = [
            current_pos[0],
            current_pos[1],
            current_pos[2] + movement_distance,
            current_pos[3],
            current_pos[4],
            current_pos[5]
        ]
        
        logger.info("Moving up 10cm")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_up[0], y=target_up[1], z=target_up[2],
            roll=target_up[3], pitch=target_up[4], yaw=target_up[5],
            speed=60, wait=True
        )
        
        if not success:
            logger.error("Up movement failed")
            return False
        
        # Pause for 1 second
        import time
        time.sleep(1)
        
        # Move down 10cm (back to original position)
        logger.info("Moving down 10cm")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=current_pos[0], y=current_pos[1], z=current_pos[2],
            roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
            speed=60, wait=True
        )
        
        if success:
            logger.info("Bidirectional vertical movement completed successfully")
        else:
            logger.error("Down movement failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in bidirectional vertical movement example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def vertical_movement_zigzag_example():
    """Example: Move robot in zigzag pattern vertically"""
    
    logger.info("=== Vertical Zigzag Movement Example ===")
    
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
        
        step_distance = 50  # 50mm steps
        steps = 4  # Total movement: 200mm = 20cm
        
        # Zigzag pattern: up, down, up, down
        positions = []
        for i in range(steps):
            if i % 2 == 0:  # Even steps: move up
                z_offset = (i // 2 + 1) * step_distance
            else:  # Odd steps: move down
                z_offset = (i // 2) * step_distance
            
            positions.append([
                current_pos[0],
                current_pos[1],
                current_pos[2] + z_offset,
                current_pos[3],
                current_pos[4],
                current_pos[5]
            ])
        
        # Execute zigzag movement
        for i, pos in enumerate(positions):
            logger.info(f"Zigzag step {i+1}: z={pos[2]}")
            success = controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=pos[0], y=pos[1], z=pos[2],
                roll=pos[3], pitch=pos[4], yaw=pos[5],
                speed=50, wait=True
            )
            
            if not success:
                logger.error(f"Zigzag step {i+1} failed")
                return False
            
            # Pause between steps
            import time
            time.sleep(0.5)
        
        # Return to original position
        logger.info("Returning to original position")
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=current_pos[0], y=current_pos[1], z=current_pos[2],
            roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
            speed=50, wait=True
        )
        
        if success:
            logger.info("Vertical zigzag movement completed successfully")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in vertical zigzag movement example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


def vertical_movement_with_rotation_example():
    """Example: Move robot vertically while rotating"""
    
    logger.info("=== Vertical Movement with Rotation Example ===")
    
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
        
        # Define combined movement (up 10cm + rotate 30 degrees)
        target_x = current_pos[0]
        target_y = current_pos[1]
        target_z = current_pos[2] + 100  # +100mm = +10cm
        target_roll = current_pos[3]
        target_pitch = current_pos[4]
        target_yaw = current_pos[5] + 30  # +30 degrees
        
        logger.info(f"Moving up and rotating to: z={target_z}, yaw={target_yaw}°")
        
        # Execute motion with data collection
        success = controller.execute_motion_with_data_collection(
            controller.move_linear,
            x=target_x, y=target_y, z=target_z,
            roll=target_roll, pitch=target_pitch, yaw=target_yaw,
            speed=60, wait=True
        )
        
        if success:
            logger.info("Vertical movement with rotation completed successfully")
            
            # Return to original position and orientation
            logger.info("Returning to original position and orientation")
            controller.execute_motion_with_data_collection(
                controller.move_linear,
                x=current_pos[0], y=current_pos[1], z=current_pos[2],
                roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5],
                speed=60, wait=True
            )
        else:
            logger.error("Vertical movement with rotation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in vertical movement with rotation example: {e}")
        return False
    
    finally:
        # Cleanup
        controller.disconnect()


if __name__ == "__main__":
    print("Vertical Movement Examples")
    print("1. Move up 10cm")
    print("2. Move down 10cm")
    print("3. Bidirectional vertical movement")
    print("4. Vertical zigzag movement")
    print("5. Vertical movement with rotation")
    
    choice = input("Select example (1-5): ").strip()
    
    if choice == "1":
        vertical_movement_up_example()
    elif choice == "2":
        vertical_movement_down_example()
    elif choice == "3":
        vertical_movement_bidirectional_example()
    elif choice == "4":
        vertical_movement_zigzag_example()
    elif choice == "5":
        vertical_movement_with_rotation_example()
    else:
        print("Invalid choice")
        vertical_movement_up_example()  # Default to first example