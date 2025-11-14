# ✋ Real-Time Hand Gesture Recognition & Finger Counter

A computer vision application that uses **MediaPipe** and **OpenCV** to detect hand landmarks in real-time and count the number of raised fingers.

This project serves as a foundational step towards building human-computer interaction (HCI) systems, such as gesture-based controls for robotics or smart devices.



## 🌟 Features

- **Real-Time Hand Tracking:** Detects and tracks 21 keypoints (landmarks) of the hand via webcam.
- **Robust Finger Counting Logic:** Accurately counts raised fingers (0-5).
- **Left/Right Hand Distinction:** Includes logic to differentiate between left and right hands for accurate thumb detection.
- **Visual Feedback:** Renders the hand skeleton and displays the final finger count directly on the video feed.

## 🛠️ Tech Stack

- **Language:** Python
- **Core Libraries:** OpenCV, MediaPipe
- **Application:** Real-time video processing

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Armin-ar79/Hand-Gesture-Recognition.git](https://github.com/Armin-ar79/Hand-Gesture-Recognition.git)
    cd Hand-Gesture-Recognition
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    py finger_counter.py
    ```

4.  Show your hand to the webcam and press 'q' to quit.
