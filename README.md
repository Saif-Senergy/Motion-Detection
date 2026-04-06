1. Overview
This document explains how to build and run an advanced motion detection and AI person detection system using Python. The system includes a graphical user interface, automatic recording, snapshots, and AI-based human detection using YOLO.
2. Features

• Live camera preview
• Motion detection using background subtraction
• AI person detection using YOLOv8
• Automatic video recording when motion/person detected
• Manual recording button
• Snapshot capture
• Adjustable sensitivity and recording duration
• Recording browser inside the application
• Works on CPU-only laptops

3. Requirements

Install Python.

Then install required libraries using:

pip install opencv-python pillow numpy ultralytics

4. How the System Works
The system captures video from the webcam using OpenCV. Motion detection is performed using background subtraction. If AI detection is enabled, a YOLOv8 model analyzes each frame to detect people. When motion or a person is detected, the system automatically records video.

5. Step-by-Step Setup

1. Install Python.
2. Install required libraries.
3. Save the provided Python code as:

ultimate_motion_detector_v2.py

4. Open Command Prompt in the project folder.
5. Run the program using:

python ultimate_motion_detector_v2.py

6. The camera window will open with the application interface.

6. User Interface Controls

Min Motion Area:
Controls how large a moving object must be before it triggers detection.

Threshold:
Controls motion sensitivity.

Record Seconds:
How long recording continues after motion stops.

Enable AI Person Detection:
Activates YOLO-based human detection.

Snapshot Button:
Saves a still image to the snapshots folder.

Manual Record Button:
Starts/stops recording manually.

Browse Recordings:
Shows saved recordings.

7. Folder Structure

project_folder
│
├── ultimate_motion_detector_v2.py
├── recordings
│   └── saved video files
└── snapshots
    └── captured images
