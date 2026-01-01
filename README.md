# Hand-Tracking Virtual Sketchpad
A real-time virtual drawing application that uses computer vision hand-tracking to allow users to draw, erase, and adjust colors and brush sizes with hand gestures.

This project uses MediaPipe's hand landmark detection and OpenCV to create an interactive sketching interface controlled by hand movements captured by the webcam.

## Features
* Real-time drawing with your index finger
* Adjustable sliders for color and line thickness
* Eraser functionality
* Hand skeleton overlay for better visibility of your fingers
* Image export options, including:
  * PNG with background
  * Transparent PNG using an alpha channel mask

## How It Works
* MediaPipe detects 21 hand landmarks per frame
* Distances between different landmarks distinguish drawing gestures from navigation gestures
* Drawing operations are performed on a separate sketch and combined with the camera feed using a binary mask
* Color and thickness controls are implemented as interactive UI elements within the camera frame
* Drawings can be saved with or without the stream from the webcam

## Installation
**Python version:** 3.10.* required

```
git clone https://github.com/ella84211/virtual-sketchpad.git
cd virtual-sketchpad
pip install -r requirements.txt
```

## Running
`python sketch.py`

## Usage
Draw with your index finger. Hold your hand as if to gesture the number '1'.
Holding another finger close to the index finger will allow you to move your hand without drawing.

You can drag the sliders along the top and left sides of the screen to change the color and line thickness.
To do this, pinch the slider and move it. Make sure your hand stays inside the frame.

Dragging the color slider into the white box with an 'X' on the right side of the color strip will turn your finger into an eraser.

Make sure to keep your whole hand inside the frame!

Once you're done drawing, here are your options:
* Press 'f' to save the image as a .png that includes your background.
* Press 's' to save the image as a .png that does not include your background.
* Press 'c' to clear the sketch.
* Press 'esc' to exit.
* Pressing 'f' or 's' will prompt you for an image name.
