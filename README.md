# EVA_CAM - Interactive Camera Control System

EVA_CAM is an integrated system that combines xArm robot control with AlpLib HVS data acquisition. The system provides real-time interactive camera control with synchronized motion and data collection capabilities.

## Features

- Real-time Robot Control: Control xArm robotic arm in real-time via keyboard
- High-speed Vision Acquisition: Support for AlpLib HVS high-speed camera data acquisition
- Interactive Operation: Intuitive keyboard control interface
- Data Synchronization: Precise synchronization between motion and data collection
- High Performance: 1000Hz update rate with smooth continuous movement
- Configurable: Flexible configuration system for different application scenarios

## System Requirements

### Hardware Requirements
- xArm Robotic Arm (recommended models)
- AlpLib HVS High-speed Camera System
- Computer: Windows 10/11, 8GB+ RAM

### Software Dependencies
- Python 3.8+
- xArm SDK
- AlpLib library files
- OpenCV
- NumPy

## Installation and Configuration

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd EVA_CAM

# Install Python dependencies
pip install opencv-python numpy keyboard
```

### 2. Hardware Connection
- Connect xArm robotic arm to network (default IP: 192.168.1.222)
- Install and configure AlpLib HVS camera
- Ensure all devices have proper network connectivity

### 3. Configuration File
Edit `eva_cam/config/settings.conf` file:

```ini
# xArm Robot Configuration
[XARM]
ip = 192.168.1.222
port = 3000
timeout = 10
default_speed = 100

# Camera Configuration
[CAMERA]
mode = HVS
aps_mode = NORMAL_V2
evs_mode = NORMAL_V2

# Motion Parameters
[MOTION]
linear_speed = 100
angular_speed = 50
rotation_speed = 30
```

## Usage

### Start the System
```bash
cd eva_cam
python interactive_camera_control.py
```

### Control Instructions
After system startup, use the following keys for control:

| Key | Function | Description |
|------|------|------|
| **A** | Move Right | Hold for continuous horizontal right movement |
| **D** | Move Left | Hold for continuous horizontal left movement |
| **W** | Move Up | Hold for continuous vertical upward movement |
| **S** | Move Down | Hold for continuous vertical downward movement |
| **Q** | Rotate CCW | Hold for continuous counter-clockwise rotation |
| **E** | Rotate CW | Hold for continuous clockwise rotation |
| **ESC** | Exit Program | Safe system exit |

### Movement Parameters
- **Movement Step**: 5mm
- **Rotation Step**: 2 degrees
- **Update Rate**: 1000Hz (1ms intervals)
- **Control Method**: State tracking with continuous movement support

## Project Structure

```
EVA_CAM/
├── eva_cam/
│   ├── interactive_camera_control.py    # Main program entry
│   ├── eva_cam_controller.py            # Core controller
│   ├── movement_horizontal.py           # Horizontal movement module
│   ├── movement_vertical.py             # Vertical movement module
│   ├── movement_rotation.py             # Rotation movement module
│   ├── utils/                           # Utility modules
│   │   ├── config.py                    # Configuration management
│   │   ├── logger.py                    # Logging system
│   │   ├── helpers.py                   # Helper functions
│   │   └── __init__.py                  # Package initialization
│   ├── config/
│   │   └── settings.conf                # Configuration file
│   └── logs/                            # Log files directory
├── CLAUDE.md                            # Claude Code guidance document
└── README.md                            # Project documentation
```

## Core Modules

### InteractiveCameraControl
- Main application class
- Handles keyboard input and state management
- Implements continuous movement mechanism

### EvaCamController
- System core controller
- Integrates robot and camera systems
- Manages data synchronization and recording

### Movement Control Modules
- Three independent movement control modules
- Support horizontal, vertical, and rotational movement
- Built-in safety checks and error handling

## Safety Features

- **Collision Detection**: Adjustable collision sensitivity
- **Emergency Stop**: ESC key for emergency exit
- **Error Handling**: Comprehensive exception handling mechanism
- **Motion Limits**: Configurable movement range and speed limits
- **Logging**: Detailed operation logging and recording

## Troubleshooting

### Common Issues

1. **Cannot Connect to xArm**
   - Check network connection
   - Verify IP address configuration
   - Confirm xArm power status

2. **Camera Initialization Failed**
   - Check AlpLib library file paths
   - Confirm camera driver installation
   - Verify camera connection status

3. **Keys Not Responding**
   - Check keyboard permissions
   - Confirm application focus
   - Review error logs

### Log Files
- Log file location: `eva_cam/logs/`
- Log file name format: `eva_cam_YYYYMMDD.log`
- Contains detailed system operation information and error diagnostics

## Development Notes

### Extending Functionality
- Add new movement modes in movement modules
- Support custom camera configurations and data acquisition parameters
- Extend keyboard control functionality

### Performance Optimization
- Movement parameters can be adjusted according to specific needs
- Data acquisition frequency is configurable
- Support for multi-threaded concurrent processing

## License

[Add license information]

## Contact

[Add contact information]

---

**Note**: Please ensure all hardware devices are properly connected and configured before first use. It is recommended to test in a safe environment.