# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AMR (Autonomous Mobile Robot) vision-based box picking and stacking system. It integrates:
- **Fairino robotic arm** (6-DOF manipulator) for pick-and-place operations
- **Intel RealSense D405** depth camera for 3D vision
- **YOLO OBB** (Oriented Bounding Box) model for box detection
- **MSIS Phoebus AMR** platform for mobile base control
- **ArUco markers** for table height calibration (optional)

The system automatically detects boxes on tables, calculates 3D positions, verifies IK feasibility, picks boxes, and stacks them at designated locations.

## Running the System

### CLI Mode (Terminal)
```bash
python main.py
```
Interactive menu with commands 0-13 for robot control, vision, IK checking, and automated pick-place cycles.

### GUI Mode (Tkinter)
```bash
python main_gui.py
```
Provides a graphical interface with buttons for all robot operations, real-time status display, and box counting.

### Demo GUI (Testing)
```bash
python demo_gui.py
```
Simplified GUI for testing specific features.

## Architecture Overview

### Core Component Flow
1. **Vision Pipeline** (`vision/box_detector.py`):
   - Captures RGB-D frames from RealSense D405
   - Runs YOLO OBB inference to detect boxes
   - Calculates 3D positions (X, Y, Z) and rotation angles
   - Filters noise using jump detection and MAD thresholds
   - Counts stacked boxes using depth-based heuristics
   - Supports ArUco marker-based table calibration

2. **Robot Control Layer** (`robot/` directory):
   - `robot_connector.py`: Manages Fairino SDK connection
   - `robot_state.py`: Caches current pose/joint readings
   - `robot_motion.py`: Executes MoveJ/MoveCart commands
   - `gripper_controller.py`: Controls parallel gripper
   - `target_pose.py`: Transforms vision coords to robot base frame
   - `ik_checker.py`: Verifies IK solutions with orientation search
   - `pick_place.py`: Executes pick operations with 2-phase approach
   - `auto_pick_place.py`: Orchestrates N-cycle automated workflows
   - `return_home.py`: Safe home position return

3. **AMR Integration** (`amr/` directory):
   - `msis_amr_motion.py`: REST API client for MSIS Phoebus AMR
   - `rest_api.py`: HTTP request wrapper
   - Supports go_to, rotate, speed control, power management

4. **Configuration** (`config/` directory):
   - `app_config.py`: Robot IP and SDK paths
   - `robot_config.py`: Offset calibration, speed, IK search, gripper, stacking waypoints
   - `vision_config.py`: YOLO model, camera settings, depth filters, ArUco, stack counting
   - `gui_config.py`: GUI layout and styling

### Key Design Patterns

**Module Dependency Injection**: All modules accept instances of their dependencies (e.g., `AutoPickPlace` receives `robot_connector`, `box_detector`, `pick_place`, etc.). This enables clean separation and testability.

**Callback-based Reconnection**: Most robot operations accept a `reconnect_cb` parameter to handle disconnections gracefully without crashing the workflow.

**Cache-based Workflow**: `cmd_*` methods save results to internal caches (e.g., `_last_measure_avg`, `_last_target_pose6`), allowing subsequent steps to use cached data without re-reading.

**2-Phase Pick Approach**:
1. **Phase 0**: Move to safe approach position (100mm above target)
2. **Target**: Lower to grasp position with precise alignment
This prevents collisions with stacked boxes.

**IK Search Strategy**: When IK fails, the system iterates through `SEARCH_RZ_LIST`, `SEARCH_RX_LIST`, `SEARCH_RY_LIST` to find valid orientation offsets (typically ±1-3 degrees).

**Vertical Box Detection**: Boxes with aspect ratio > `MAX_ASPECT_RATIO` (4.2) are flagged as vertical. If `ENABLE_VERTICAL_BOX_ROTATION=True`, the system rotates them horizontal using J5 tilt; otherwise they're excluded.

## Configuration Deep Dive

### Critical Calibration Parameters (`robot_config.py`)

```python
# Camera-to-gripper offset (measured in robot base frame)
OFF_X_MM = -10.0   # Left/right offset
OFF_Y_MM = -70.0   # Forward/back offset
OFF_Z_MM = -165.0  # Vertical offset (camera above gripper)
PIVOT_LENGTH = 175.0  # Gripper pivot distance

# Base frame rotation (compensates for robot mounting angle)
BASE_YAW_OFFSET_DEG = 90

# Stack locations (JOINT positions for collision-free paths)
WP11_A_JOINT = [-66.404, -93.019, -50.178, -126.227, 90.512, 116.450]  # Position 1
WP11_B_A_JOINT = [-88.803, -93.717, -51.486, -126.244, 89.066, 91.287]  # Position 2
WP11_DROP_BASE_POSE = [92.489, -431.770, 247.020, -177.935, -0.393, 88.608]  # Drop pose 1
```

### Vision Tuning (`vision_config.py`)

```python
# YOLO inference
CONF_THRES = 0.85  # Higher = fewer false positives
IOU_THRES = 0.85   # Higher = more aggressive NMS

# Depth filtering
ROI_MARGIN_PX = 6.0       # Shrink OBB inward to avoid edges
MIN_ROI_PIXELS = 120      # Minimum valid depth pixels
MAD_THRES_M = 0.020       # Median Absolute Deviation threshold (20mm)

# Jump filter (detects sudden position changes)
JUMP_XY_MM = 35.0         # Max XY movement between frames
JUMP_Z_MM = 60.0          # Max Z movement between frames
MAX_CONSECUTIVE_JUMPS = 3 # After 3 jumps, assume box moved

# Stack counting
BOX_HEIGHT_MM = 58.0      # Height of one box
ARUCO_MARKER_BASELINES = {
    1: 670.0,   # Low table
    2: 860.0,   # Medium table
    3: 1000.0,  # High table
}
```

## Common Development Tasks

### Adjusting Pick-Place Speed
Edit `robot_config.py`:
```python
MOVE_CART_VEL_PHASE0 = [100, 70, 50, 30]  # Approach speed (higher = faster)
MOVE_CART_VEL_TARGET = [30, 20, 10]       # Grasp descent speed
MOVEJ_VEL = 100                           # Joint move speed (0-180)
```

### Recalibrating Camera Offset
1. Manually jog robot to align gripper with a known object
2. Read vision measurement (`cmd_measure_avg`)
3. Calculate actual gripper position
4. Update `OFF_X_MM`, `OFF_Y_MM`, `OFF_Z_MM` in `robot_config.py`

### Adding New Stack Positions
1. Teach waypoint in joint mode (to avoid singularities)
2. Record JOINT coordinates (not POSE)
3. Add to `robot_config.py` as `WP11_C_A_JOINT` and `WP11_C_DROP_BASE_POSE`
4. Update `auto_pick_place.py` to handle additional positions

### Debugging IK Failures
Check `robot/ik_checker.py`:
- Increase `SEARCH_TIMEOUT_SEC` if searches timeout
- Add more candidates to `SEARCH_RZ_LIST` (e.g., ±40, ±50)
- Verify target Z is within robot reach (150-800mm typical)

### Testing Vision Without Robot
```bash
python test/test_box_detector_standalone.py
```
Runs vision loop and displays detections in preview window.

## System Startup Checklist

1. **Hardware**: Power on Fairino robot, connect RealSense D405, ensure AMR is awake
2. **Network**: Verify robot IP reachable (`192.168.0.15` default)
3. **Model**: Ensure YOLO model exists at path in `vision_config.py`
4. **Home Position**: Move robot to safe home position before starting
5. **Launch**: Run `main_gui.py` or `main.py`
6. **Initialize**: Press "0" (Connect) → "1" (Vision ON) → "2" (Save Home)
7. **Test**: Press "3" (Measure) to verify vision detects boxes

## Quick Start (13번 기능)

The system includes a one-button automated workflow:
```
Command 13: Quick Start
```
This automatically:
1. Connects to robot
2. Starts vision system
3. Saves current position as home
4. Executes 4-box pick-place cycles
5. Handles vertical box rotation (if enabled)
6. Updates stack counter

Equivalent to: `0 → 1 → 2 → (12번 with 4 cycles)`

## Coordinate Systems

### Vision Frame (Camera)
- Origin: Camera optical center
- X: Right (positive)
- Y: Down (positive)
- Z: Forward into scene (positive)

### Robot Base Frame
- Origin: Robot base center
- X: Forward (positive)
- Y: Left (positive)
- Z: Up (positive)
- RZ: Rotation around Z-axis (yaw)

### Transformation Pipeline
```
Vision XYZ → Camera offset → Base yaw rotation → Robot base XYZ
```
See `target_pose.py:build_target_pose6()` for implementation.

## Threading and Safety

- **Vision Loop**: Runs in background thread (`_detection_loop`)
- **Robot Commands**: Blocking in main thread (prevents command overlap)
- **Reconnection**: `reconnect_cb` parameter handles transient disconnections
- **Emergency Stop**: GUI "Stop" button calls `robot.StopMotion()`
- **Vision Restart Lock**: During robot motion, vision restart is disabled to prevent RealSense crashes

## Important Constraints

1. **IK Singularities**: Near-vertical targets (RY > 5°) often fail IK
2. **Depth Range**: RealSense D405 works best at 150-800mm
3. **Box Orientation**: System assumes boxes lie flat (not tilted > 10°)
4. **Stack Limit**: Maximum 6 boxes (3 per position) due to reach constraints
5. **Speed Fallback**: If high-speed motion fails, system retries at lower speeds
6. **Gripper Settling**: 100ms delay after open/close ensures stable grip

## File Organization

- `main.py`, `main_gui.py`: Entry points
- `robot/`: All robot control modules (8 files)
- `vision/`: Vision pipeline (2 files + utils)
- `amr/`: Mobile base REST API client (3 files)
- `gui/`: GUI components (2 files)
- `config/`: All configuration (4 files)
- `test/`: Standalone test scripts
- `old_code/`: Archived implementations

## SDK Dependencies

- **Fairino Python SDK**: Located at `FAIRINO_PYD_PATH` (Linux .so or Windows .pyd)
  - Must be added to `sys.path` before importing `Robot` module
  - Requires exact Python version match (3.10 for Jetson)
- **pyrealsense2**: Intel RealSense SDK wrapper
- **ultralytics**: YOLO inference engine
- **opencv-python**: Image processing
- **numpy**: Array operations

## Common Issues

**"Frame didn't arrive"**: RealSense timeout, usually from USB bandwidth. Reduce FPS or resolution.

**IK search exhausted**: Target unreachable. Check Z height, verify not too far/close, try adjusting `SEARCH_RZ_LIST`.

**Vision detects no boxes**: Check lighting, CONF_THRES too high, or box outside DEPTH_MIN_M/DEPTH_MAX_M range.

**Robot disconnects mid-cycle**: Network instability. Verify cable connection, reduce speed to lower vibration.

**Vertical boxes not rotating**: Set `ENABLE_VERTICAL_BOX_ROTATION = True` in `vision_config.py`.

**Stack count wrong**: Recalibrate `BASELINE_DEPTH_*_MM` or use ArUco markers for auto-calibration.

## Performance Tuning

- **Increase speed**: Raise `MOVEJ_VEL` and `MOVE_CART_VEL_*` in `robot_config.py`
- **Reduce latency**: Lower `AVG_N` (10 → 5) in `vision_config.py`
- **Improve accuracy**: Lower `MOVE_CART_VEL_TARGET` (30 → 20)
- **Parallelize**: Move vision measurement during robot return home (already implemented)
