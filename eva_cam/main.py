#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA_CAM main entry point
Provides a command-line interface for running different motion examples
"""

import sys
import os
import argparse
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import logger
from eva_cam_controller import EvaCamController


def run_example(example_name: str, config_path: Optional[str] = None):
    """Run a specific example"""
    
    logger.info(f"Running example: {example_name}")
    
    try:
        if example_name == "horizontal":
            from movement_horizontal import horizontal_movement_example
            horizontal_movement_example()
        
        elif example_name == "horizontal_bidirectional":
            from movement_horizontal import horizontal_movement_bidirectional_example
            horizontal_movement_bidirectional_example()
        
        elif example_name == "horizontal_y":
            from movement_horizontal import horizontal_movement_y_direction_example
            horizontal_movement_y_direction_example()
        
        elif example_name == "rotation":
            from movement_rotation import rotation_yaw_example
            rotation_yaw_example()
        
        elif example_name == "rotation_bidirectional":
            from movement_rotation import rotation_bidirectional_example
            rotation_bidirectional_example()
        
        elif example_name == "rotation_roll":
            from movement_rotation import rotation_roll_example
            rotation_roll_example()
        
        elif example_name == "rotation_pitch":
            from movement_rotation import rotation_pitch_example
            rotation_pitch_example()
        
        elif example_name == "rotation_combined":
            from movement_rotation import rotation_combined_example
            rotation_combined_example()
        
        elif example_name == "vertical_up":
            from movement_vertical import vertical_movement_up_example
            vertical_movement_up_example()
        
        elif example_name == "vertical_down":
            from movement_vertical import vertical_movement_down_example
            vertical_movement_down_example()
        
        elif example_name == "vertical_bidirectional":
            from movement_vertical import vertical_movement_bidirectional_example
            vertical_movement_bidirectional_example()
        
        elif example_name == "vertical_zigzag":
            from movement_vertical import vertical_movement_zigzag_example
            vertical_movement_zigzag_example()
        
        elif example_name == "vertical_with_rotation":
            from movement_vertical import vertical_movement_with_rotation_example
            vertical_movement_with_rotation_example()
        
        elif example_name == "custom_template":
            from custom_motion_template import create_custom_sequence_example
            create_custom_sequence_example()
        
        else:
            logger.error(f"Unknown example: {example_name}")
            print_examples()
            return False
        
        logger.info(f"Example '{example_name}' completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running example '{example_name}': {e}")
        return False


def print_examples():
    """Print available examples"""
    print("\nAvailable Examples:")
    print("Horizontal Movement:")
    print("  horizontal                    - Move 10cm horizontally (X direction)")
    print("  horizontal_bidirectional      - Move 10cm forward and backward")
    print("  horizontal_y                  - Move 10cm in Y direction")
    
    print("\nRotation Movement:")
    print("  rotation                      - Rotate 30° around yaw axis")
    print("  rotation_bidirectional        - Rotate 30° clockwise and counter-clockwise")
    print("  rotation_roll                 - Rotate 30° around roll axis")
    print("  rotation_pitch                - Rotate 30° around pitch axis")
    print("  rotation_combined             - Combined rotation (15° each axis)")
    
    print("\nVertical Movement:")
    print("  vertical_up                   - Move 10cm up")
    print("  vertical_down                 - Move 10cm down")
    print("  vertical_bidirectional        - Move 10cm up and down")
    print("  vertical_zigzag               - Zigzag vertical movement")
    print("  vertical_with_rotation        - Vertical movement with rotation")
    
    print("\nCustom Motion:")
    print("  custom_template               - Custom motion sequence example")
    
    print("\nInteractive Mode:")
    print("  interactive                   - Interactive mode for custom motion creation")


def interactive_mode():
    """Interactive mode for creating custom motions"""
    print("=== EVA_CAM Interactive Mode ===")
    
    controller = EvaCamController()
    
    try:
        # Initialize systems
        if not controller.initialize_xarm():
            logger.error("Failed to initialize xArm")
            return
        
        if not controller.initialize_camera():
            logger.error("Failed to initialize camera")
            return
        
        from custom_motion_template import CustomMotionBuilder
        builder = CustomMotionBuilder(controller)
        
        print("\nAvailable motion patterns:")
        patterns = builder.list_patterns()
        for i, pattern in enumerate(patterns, 1):
            info = builder.get_pattern_info(pattern)
            print(f"  {i}. {pattern} - {info['description']}")
        
        while True:
            print("\nOptions:")
            print("  1. Execute single pattern")
            print("  2. Create custom sequence")
            print("  3. Get system status")
            print("  4. Emergency stop")
            print("  5. Exit")
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                # Execute single pattern
                pattern_idx = int(input(f"Select pattern (1-{len(patterns)}): ")) - 1
                if 0 <= pattern_idx < len(patterns):
                    pattern_name = patterns[pattern_idx]
                    info = builder.get_pattern_info(pattern_name)
                    print(f"\nPattern: {pattern_name}")
                    print(f"Description: {info['description']}")
                    print("Default parameters:")
                    for key, value in info['parameters'].items():
                        print(f"  {key}: {value}")
                    
                    use_default = input("Use default parameters? (y/n): ").strip().lower()
                    if use_default == 'n':
                        print("Enter custom parameters:")
                        params = {}
                        for key, default_value in info['parameters'].items():
                            user_input = input(f"{key} [{default_value}]: ").strip()
                            if user_input:
                                # Try to convert to appropriate type
                                try:
                                    if isinstance(default_value, bool):
                                        params[key] = user_input.lower() in ['true', 'yes', '1', 'y']
                                    elif isinstance(default_value, int):
                                        params[key] = int(user_input)
                                    elif isinstance(default_value, float):
                                        params[key] = float(user_input)
                                    else:
                                        params[key] = user_input
                                except ValueError:
                                    params[key] = default_value
                            else:
                                params[key] = default_value
                    else:
                        params = None
                    
                    builder.execute_pattern(pattern_name, params)
                else:
                    print("Invalid pattern selection")
            
            elif choice == "2":
                # Create custom sequence
                print("Create custom sequence (enter 'done' when finished)")
                sequence = []
                
                while True:
                    pattern_input = input("Add pattern (name or 'done'): ").strip()
                    if pattern_input.lower() == 'done':
                        break
                    
                    if pattern_input in patterns:
                        info = builder.get_pattern_info(pattern_input)
                        print(f"Pattern: {pattern_input}")
                        print("Default parameters:")
                        for key, value in info['parameters'].items():
                            print(f"  {key}: {value}")
                        
                        use_default = input("Use default parameters? (y/n): ").strip().lower()
                        if use_default == 'n':
                            params = {}
                            for key, default_value in info['parameters'].items():
                                user_input = input(f"{key} [{default_value}]: ").strip()
                                if user_input:
                                    try:
                                        if isinstance(default_value, bool):
                                            params[key] = user_input.lower() in ['true', 'yes', '1', 'y']
                                        elif isinstance(default_value, int):
                                            params[key] = int(user_input)
                                        elif isinstance(default_value, float):
                                            params[key] = float(user_input)
                                        else:
                                            params[key] = user_input
                                    except ValueError:
                                        params[key] = default_value
                                else:
                                    params[key] = default_value
                        else:
                            params = None
                        
                        sequence.append((pattern_input, params))
                    else:
                        print(f"Unknown pattern: {pattern_input}")
                
                if sequence:
                    print(f"\nExecuting sequence with {len(sequence)} patterns...")
                    builder.execute_sequence(sequence)
                else:
                    print("No patterns in sequence")
            
            elif choice == "3":
                # Get system status
                status = controller.get_system_status()
                print("\nSystem Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
            
            elif choice == "4":
                # Emergency stop
                confirm = input("Are you sure you want to trigger emergency stop? (y/n): ").strip().lower()
                if confirm == 'y':
                    controller.emergency_stop()
                    print("Emergency stop triggered!")
            
            elif choice == "5":
                # Exit
                print("Exiting interactive mode...")
                break
            
            else:
                print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
    
    finally:
        controller.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='EVA_CAM - Integrated xArm and AlpLib Control System')
    parser.add_argument('example', nargs='?', help='Example to run (use "list" to see examples)')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--list', '-l', action='store_true', help='List available examples')
    
    args = parser.parse_args()
    
    if args.list:
        print_examples()
        return
    
    if args.interactive:
        interactive_mode()
        return
    
    if args.example == "list":
        print_examples()
        return
    
    if args.example:
        run_example(args.example, args.config)
    else:
        print("EVA_CAM - Integrated xArm and AlpLib Control System")
        print("\nUsage:")
        print("  python main.py <example>          # Run specific example")
        print("  python main.py --list             # List available examples")
        print("  python main.py --interactive      # Run in interactive mode")
        print("  python main.py --config <path>   # Use custom configuration")
        print("\nExamples:")
        print_examples()


if __name__ == "__main__":
    main()