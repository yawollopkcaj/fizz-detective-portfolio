# Controller Package (ROS)

Runtime ROS package controlling the autonomous robot during the Fizz Detective competition.

## Contents
- `scripts/lane_follower_bc.py` — Behavior Cloning policy inference  
- `scripts/plate_detector_yolo.py` — optional YOLO plate detector  
- `scripts/plate_ocr_infer.py` — OCR model inference  
- `scripts/score_client.py` — start/stop and clue submission  
- `scripts/respawn_helper.py` — service caller for pink-line respawns  
- `cfg/params.yaml` — all tuning constants (v / ω limits, thresholds)  
- `models/` — symlink or submodule to trained models  

## Run
```bash
roslaunch controller_pkg controller.launch team:=TeamRed pass:=multi21
```

## Node Graph
`camera → lane_follower_bc → cmd_vel`  
`lane_follower_bc → score_client → /score_tracker`

## Development
- Publish/sub to only allowed topics:
  - Sub: `/B1/pi_camera/image_raw`, `/clock`
  - Pub: `/B1/cmd_vel`, `/score_tracker`
- Add 1 s startup delay before first message.
- Rate-limit command publishing (≈ 10–15 Hz).

## Notes
- Load models via `onnxruntime` or TensorRT if available.  
- Ensure `models/bc/meta.json` and `models/ocr/meta.json` match exported versions.  
- Tested with ROS Noetic + Gazebo 11.
