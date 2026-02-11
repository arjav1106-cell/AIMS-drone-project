## AIMS-drone-project
this repository is made by Arjav Jain 25/B01/033 for the 'hand gesture drone control' project

### NOTE:
#### I have made some commits and updated the README.md file 'after' the submission time because that time there was less time available to me and I apologise for that. I the process of uploading the newer model i have learned and changed a few things. Once again, I am sorry for the delay!

# 🖐️ Drone Hand Gesture Control System

Real-time hand gesture recognition using **MediaPipe** and a **CNN model** with webcam input to control a drone.

---

## 🚀 Features

- 🎥 Live gesture detection via webcam  
- ✋ MediaPipe hand landmark extraction  
- 🧠 CNN-based gesture classification (shape recognition)  
- 🧭 Direction logic using finger geometry (1 finger / 2 fingers)  
- 🛑 Safety system: STOP, EMERGENCY, EXIT EMERGENCY  
- 🔁 Temporal smoothing + confidence threshold for stable control  
- 🧱 State machine to avoid repeated commands  

---

## ✋ Supported Gestures (for Media Pipe)

| Gesture | Meaning / Use |
|--------|---------------|
| ✋ Open Palm | STOP |
| ✊ Fist | LAND |
| ✋ One Hand Open | TAKEOFF |
| 👉 One Finger Right | RIGHT |
| 👈 One Finger Left | LEFT |
| 👆 One Finger Up | UP |
| 👇 One Finger Down | DOWN |
| ✌️ Two Fingers Up | UP_LEFT |
| ✌️ Two Fingers Left | DOWN_LEFT |
| ✌️ Two Fingers Down | DOWN_RIGHT |
| ✌️ Two Fingers Right | UP_RIGHT |
| 🤏 Thumb + Index | ROTATE_CW |
| 🤘 Thumb + Index + Pinky | ROTATE_CCW |
| ✊✊ Two Hands Fists | EMERGENCY |
| ✋✋ Two Hands Open | EXIT EMERGENCY |

---

## ✋ Supported Gestures (for CNN)

| Gesture | Meaning / Use |
|--------|---------------|
| ✋ Open Palm | STOP |
| ✊ Fist | LAND |
| 👍 Thumb Up | TAKEOFF |
| 👉 One Finger Right | RIGHT |
| 👈 One Finger Left | LEFT |
| 👆 One Finger Up | UP |
| 👇 One Finger Down | DOWN |
| ✌️ Two Fingers Up | UP_LEFT |
| ✌️ Two Fingers Left | DOWN_LEFT |
| ✌️ Two Fingers Down | DOWN_RIGHT |
| ✌️ Two Fingers Right | UP_RIGHT |
| 🤏 Thumb + Index | ROTATE_CW |
| 🤘 Thumb + Index + Pinky | ROTATE_CCW |
| ✊✊ Two Hands Fists | EMERGENCY |
| ✋✋ Two Hands Open | EXIT EMERGENCY |

---

## 🧠 System Design (Flow Chart)

### 1️⃣ MediaPipe-Based Pipeline (Rule-Based, Geometry)
- Webcam Frame
- MediaPipe Hands (21 landmarks per hand)
- Finger State Detection (up/down)
- Geometric Analysis (angles, vectors, directions)
- Gesture Logic (if-else rules)
- Command Generator (UP, DOWN, LEFT, STOP, etc.)
- Drone / Control Interface


### 2️⃣ CNN-Based Pipeline (Learning-Based)

- Webcam Frame
- ROI Crop (hand region)
- Grayscale + Resize (128x128)
- CNN Model
- Softmax Probabilities
- Confidence Threshold + Temporal Smoothing
- State Machine (debounce / safety rules)
- Final Command Output

---

## 📂 Dataset Information

This project uses a custom grayscale image dataset:

- 📊 1250 training images per class  
- 📁 Each class in a separate folder  
- 🖼️ Images resized to 128×128 (grayscale)  

```bash
dataset_shapes/
 ├── FIST/
 ├── ONE_FINGER/
 ├── TWO_FINGER/
 ├── OPEN_PALM/
 ├── THUMB_UP/
 ├── THUMB_INDEX/
 └── THUMB_INDEX_PINKY/
```

---

```bash
dataset/
├── DOWN/
├── DOWN_LEFT/
├── DOWN_RIGHT/
├── EMERGENCY/
├── EXIT_EMERGENCY/
├── LAND/
├── LEFT/
├── RIGHT/
├── ROTATE_CCW/
├── ROTATE_CW/
├── STOP/
├── TAKEOFF/
├── UP/
├── UP_LEFT/
└── UP_RIGHT/
```

## Run

### 1️⃣ MediaPipe-Based Pipeline
- Change the directory.
```bash
cd "C:/Git programs/AIMS/AIMS-drone-project/drone-project"
```
- Create Virtual Environment.
```bash
python -m venv .venv-mediapipe
```
- Activate it.
```bash
source .venv-mediapipe/Scripts/activate
```
- Install the Requirements.
```bash
pip install -r requirements_mp.txt
```
- Test if the install is successful.
```bash
python -c "import cv2, mediapipe, numpy; print('MediaPipe env OK')"
```
- Deactivate the environment to use the other.
```bash
deactivate
```

---

### 2️⃣ CNN-Based Pipeline
- Change the directory.
```bash
cd "C:/Git programs/AIMS/AIMS-drone-project/drone-project"
```
- Create Virtual Environment.
```bash
python -m venv .venv-cnn
```
- Activate it.
```bash
source .venv-cnn/Scripts/activate
```
- Install the Requirements.
```bash
pip install -r requirements_cnn.txt
```
```bash
python -c "import tensorflow as tf; import cv2; print('CNN env OK', tf.__version__)"
```

---

## While Switching between both the projects

### 1️⃣ MediaPipe-Based Pipeline
- Change the directory.
```bash
cd "C:/Git programs/AIMS/AIMS-drone-project/drone-project"
```
- Then Run.
```bash
source .venv-mediapipe/Scripts/activate
python media_pipe_method/main[0]_mp.py
```

---

### 2️⃣ CNN-Based Pipeline
- Change the directory.
```bash
cd "C:/Git programs/AIMS/AIMS-drone-project/drone-project"
```
- Then Run.
```bash
source .venv-cnn/Scripts/activate
python media_pipe_method/main[0]_mp.py
```

## NOTES:-
- Trained model file (`.h5`) was not included due to GitHub size limits.
- Download the trained `.h5` model file here: https://drive.google.com/drive/folders/12PE_GYuKIhXGpyu_kjfU1bHNht4VIkqc?usp=drive_link
- Place downloaded models inside the `CNN method` folder