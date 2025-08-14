# EVA_CAM Independent Version

这个独立的EVA_CAM版本包含了所有必要的依赖项，无需安装外部的AlpLib和xArm-Python-SDK。

## 目录结构

```
eva_cam_independent/
├── eva_cam/                  # 主要的EVA_CAM代码
│   ├── eva_cam_controller.py # 主控制器
│   ├── main.py              # 主入口
│   ├── movement_*.py        # 运动控制模块
│   ├── utils/               # 工具模块
│   ├── config/              # 配置文件
│   ├── data/                # 数据存储
│   └── logs/                # 日志文件
├── xarm/                    # xArm Python SDK (本地副本)
│   ├── core/                # 核心通信和工具
│   ├── tools/               # 辅助工具
│   ├── wrapper/             # 高级API包装器
│   └── x3/                  # xArm特定功能
├── alplib/                  # AlpLib (本地副本)
│   ├── bin/                 # DLL和Python模块
│   └── config/              # 相机配置文件
└── README.md               # 本文件
```

## 使用方法

1. **运行示例**:
   ```bash
   cd eva_cam_independent
   python eva_cam/main.py horizontal
   ```

2. **交互模式**:
   ```bash
   python eva_cam/main.py --interactive
   ```

3. **查看可用示例**:
   ```bash
   python eva_cam/main.py --list
   ```

## 依赖项

所有依赖项都已包含在项目中：
- **xArm控制**: 使用本地的`xarm/`目录
- **相机控制**: 使用本地的`alplib/`目录
- **Python模块**: AlpPython.pyd文件在`alplib/bin/`中

## 配置

编辑`eva_cam/config/settings.conf`文件来配置：
- xArm的IP地址
- 相机参数
- 运动参数
- 安全设置

## 注意事项

1. 确保Python版本与AlpPython.pyd文件匹配（示例中为Python 3.10）
2. 确保系统PATH中包含必要的Visual C++运行时库
3. 相机配置文件已从原始AlpLib复制到本地
4. 所有DLL文件都已包含在本地目录中

## 优势

- 无需安装外部SDK
- 可以轻松复制到其他机器
- 所有依赖项都在项目目录中
- 保持了原始EVA_CAM的所有功能

## Configuration

Edit `config/settings.conf` to match your setup:

```ini
[XARM]
ip = 192.168.1.113          # xArm IP address
default_speed = 100          # Default movement speed

[CAMERA]
mode = HVS                   # Camera mode (APS, EVS, HVS)
aps_mode = NORMAL_V2         # APS processing mode
evs_mode = NORMAL_V2         # EVS processing mode

[RECORDING]
format = hdf5                # Data format (hdf5, bin)
output_dir = ./data          # Output directory
```

## Usage

### Quick Start

```bash
# Run a horizontal movement example
python main.py horizontal

# Run in interactive mode
python main.py --interactive

# List all available examples
python main.py --list
```

### Available Examples

#### Horizontal Movement
- `horizontal` - Move 10cm horizontally (X direction)
- `horizontal_bidirectional` - Move 10cm forward and backward
- `horizontal_y` - Move 10cm in Y direction

#### Rotation Movement
- `rotation` - Rotate 30° around yaw axis
- `rotation_bidirectional` - Rotate 30° clockwise and counter-clockwise
- `rotation_roll` - Rotate 30° around roll axis
- `rotation_pitch` - Rotate 30° around pitch axis
- `rotation_combined` - Combined rotation (15° each axis)

#### Vertical Movement
- `vertical_up` - Move 10cm up
- `vertical_down` - Move 10cm down
- `vertical_bidirectional` - Move 10cm up and down
- `vertical_zigzag` - Zigzag vertical movement
- `vertical_with_rotation` - Vertical movement with rotation

#### Custom Motion
- `custom_template` - Custom motion sequence example

### Interactive Mode

The interactive mode allows you to create custom motion sequences:

```bash
python main.py --interactive
```

In interactive mode, you can:
- Execute single motion patterns with custom parameters
- Create complex motion sequences
- Monitor system status
- Trigger emergency stops

### Custom Motion Creation

Use the `custom_motion_template.py` framework to create personalized movements:

```python
from custom_motion_template import CustomMotionBuilder

# Initialize builder
builder = CustomMotionBuilder(controller)

# Create custom sequence
sequence = [
    ('linear', {'distance_x': 100, 'speed': 100}),
    ('rotational', {'rotation_yaw': 45, 'speed': 40}),
    ('circular', {'radius': 50, 'steps': 8})
]

# Execute sequence
builder.execute_sequence(sequence)
```

## Motion Pattern Types

### Linear Motion
Move in a straight line with customizable:
- Distance (X, Y, Z axes)
- Speed
- Return to start option
- Pause duration

### Rotational Motion
Rotate around specified axes with customizable:
- Rotation angles (roll, pitch, yaw)
- Speed
- Return to start option
- Pause duration

### Circular Motion
Move in circular patterns with customizable:
- Radius
- Number of steps
- Rotation axis
- Speed
- Return to start option

### Joint Motion
Control individual joints with customizable:
- Joint angles (7 joints)
- Speed
- Return to start option
- Pause duration

## Data Collection

The system automatically collects HVS data during robot motion:

- **APS Data**: High-resolution image frames
- **EVS Data**: Event-based vision data
- **Synchronization**: Data collection starts/stops with motion
- **Storage**: HDF5 or binary format with timestamps

## Safety Features

- **Emergency Stop**: Immediate halt of all motion and data collection
- **Collision Detection**: Configurable sensitivity levels
- **Parameter Validation**: Prevents invalid movements
- **System Monitoring**: Real-time status reporting

## Troubleshooting

### Common Issues

1. **Connection Issues**
   - Check IP addresses in configuration
   - Verify network connectivity
   - Ensure robot and camera are powered on

2. **Motion Failures**
   - Check robot workspace boundaries
   - Verify parameter validation
   - Look for collision warnings

3. **Data Collection Issues**
   - Check camera initialization
   - Verify storage directory permissions
   - Ensure sufficient disk space

### Logging

System logs are stored in `logs/eva_cam_YYYYMMDD.log` with detailed information about:
- System initialization
- Motion execution
- Data collection status
- Error messages

## API Reference

### EvaCamController

Main controller class for system integration:

```python
controller = EvaCamController()

# Initialize systems
controller.initialize_xarm()
controller.initialize_camera()

# Execute motion with data collection
controller.execute_motion_with_data_collection(
    controller.move_linear,
    x=100, y=0, z=0, speed=100
)

# System control
controller.emergency_stop()
controller.disconnect()
```

### Motion Methods

- `move_linear(x, y, z, roll, pitch, yaw, speed, wait)` - Linear motion
- `move_joint(angles, speed, wait)` - Joint motion
- `execute_motion_with_data_collection(motion_func, *args, **kwargs)` - Synchronized motion and data collection

## Contributing

To add new motion patterns:
1. Extend the `MotionPattern` base class
2. Implement `execute()` and `get_parameters()` methods
3. Add to the `CustomMotionBuilder` patterns dictionary
4. Update documentation and examples

## License

This project integrates open-source components. Please refer to the respective licenses of xArm-Python-SDK and AlpLib for usage terms.