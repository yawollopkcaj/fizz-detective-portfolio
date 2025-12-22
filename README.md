# Fizz Detective: Autonomous Mobile Robot (ROS & Imitation Learning)

![Robot Demo](media/fizz-detective-conpetition.gif)

## Project Overview
**Fizz Detective** is an autonomous robot software stack developed for the ENPH 353 competition. The objective was to navigate a simulated urban environment in **Gazebo**, adhering to traffic laws while identifying and reading alphanumeric "clue plates" to solve a puzzle.

The system utilizes a **Hybrid Control Architecture**, decoupling high-level decision-making (Finite State Machine) from low-level perception (End-to-End Imitation Learning) to optimize for the simulation's Real-Time Factor (RTF).

### Performance
* **Lap Time:** 2m 20s (0.8 m/s constant linear velocity).
* **Capabilities:** Lane following on unlined dirt roads, dynamic obstacle avoidance (pedestrians/trucks), and OCR character recognition.

## Source Code (Modular Architecture)
This project is split into three decoupled repositories to ensure modularity. 
* **[control](https://github.com/yawollopkcaj/fizz-detective-portfolio):** Main ROS package, Finite State Machine, and Launch files.
* **[il-training](https://github.com/yawollopkcaj/fizz-detective-il-training):** PyTorch implementation of the PilotNet Imitation Learning model.
* **[ocr-training](https://github.com/yawollopkcaj/fizz-detective-ocr-training):** YOLOv5 training pipeline and character generation scripts.

## System Architecture
The robot uses a dual-camera setup to balance inference speed with detection range:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Lane Following** | **PilotNet (PyTorch)** | End-to-End Imitation Learning CNN processing 800x800 images (resized to 128x128). |
| **Object Detection** | **YOLOv5** | Detects "Clue Boards" and Pedestrians using a dedicated 4K wide-angle stream. |
| **Decision Making** | **Finite State Machine** | Toggles perception nodes (e.g., disabling pedestrian detection while driving) to save GPU resources. |
| **OCR Pipeline** | **OpenCV + CNN** | Custom morphological pipeline (erode/dilate) to segment and read characters. |


![System Diagram](media/ENPH-353-Software-Architecture.png)

## Key Technical Challenges

### 1. The "Fat Letter" Problem (Computer Vision)
[cite_start]**Problem:** At medium distances, our HSV thresholding filter caused characters on the clue boards (like "SIZE") to bleed together into a single blob, causing the CNN to fail[cite: 239, 244].

**Solution:** We implemented a morphological preprocessing pipeline. [cite_start]We applied **erosion** to separate the connected white pixels of the characters, then calculated bounding boxes, and finally applied **dilation** to restore the character shapes before feeding them into the classification network[cite: 247, 248].

![Morphological Processing](media/debug_view.png)

### 2. Sim-to-Real Latency (RTF Variance)
**Problem:** The Imitation Learning model was trained on a local machine with a Real-Time Factor (RTF) of ~0.9. [cite_start]However, the competition server ran at ~0.55 RTF due to overhead, causing the robot to oversteer and oscillate[cite: 153, 298].

**Solution:** To mitigate this, we implemented a **Recovery Strategy** in our training data. [cite_start]We deliberately recorded "recovery maneuvers" (driving off-center and correcting sharply) to teach the model how to handle state-drift caused by lag[cite: 74].

## Neural Network Details
### PilotNet (Driving Policy)
We adapted the NVIDIA PilotNet architecture (5 convolutional layers, 3 fully connected).
* **Modification:** We strictly **removed Dropout layers**. [cite_start]While dropout usually prevents overfitting, we found that for this specific dirt-road terrain, "overfitting" to specific ground textures actually improved performance where lane lines were missing[cite: 96, 98, 99].

### Character Recognition
* [cite_start]**Training:** Synthetic data generation using affine transformations and Gaussian noise to match the low-fidelity Gazebo textures[cite: 196, 261].
* [cite_start]**Loss:** Converged after ~10 epochs using Adam optimizer[cite: 129].


![Training Loss](media/training_plot.png)

## Authors
* [cite_start]**Jack Polloway:** Driving Policy (IL), PilotNet Architecture, System Integration[cite: 12].
* [cite_start]**Ryan Mahinpey:** OCR Pipeline, YOLOv5 Implementation[cite: 14].

---
*Created for UBC Engineering Physics 353 (2025).*
