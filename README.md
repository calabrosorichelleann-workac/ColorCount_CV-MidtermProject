# ColorCount CV — Real-Time Color-Based Object Detection and Counting

**Course:** CSC 126 - Computer Vision (Midterm Project)
**Technique:** Classical OpenCV (no trained model)
**Input:** Live webcam, static image, or recorded video
**Detected colors:** Red, green, and blue

## 1. Project overview

ColorCount CV detects red, green, and blue regions in a live webcam feed, a static image, or a recorded video. It draws a bounding box and a label (color name and contour area in pixels) around each region, and shows the count per color plus the total on a dashboard. Press **M** to also see the binary color masks used for segmentation. Use case: counting or sorting color-coded objects on a table.

## 2. OpenCV topics demonstrated

1. **Image Operations** — `cv2.flip`, `cv2.resize`, `cv2.GaussianBlur`, `cv2.cvtColor` (BGR to HSV), and overlays with `cv2.rectangle`, `cv2.putText`, `cv2.addWeighted`.
2. **Image Segmentation** — `cv2.inRange` HSV masks (red uses two hue ranges), `cv2.bitwise_or`, and morphological opening/closing.
3. **Image Analysis and Transformation** — `cv2.findContours`, `cv2.contourArea` (regions under 500 px are ignored), and `cv2.boundingRect`.

## 3. Repository contents

| Path | Purpose |
|---|---|
| `colorcount_cv.py` | Application source code |
| `requirements.txt` | Python dependencies (`opencv-python`, `numpy`) |
| `sample_inputs/` | Sample files: `sample_objects.png` (6 separated objects), `sample_touching.png` (two touching red circles), `sample_video.avi` (moving objects) |
| `screenshots/` | Output screenshots (saved with the **S** key); see `screenshots/README.txt` |
| `docs/CSC 126 Documentation Project.pdf` | Project documentation paper (`.docx` source in the same folder) |
| `.gitignore` | Keeps `.venv/` and cache files out of the repository |

## 4. Requirements

- Python 3.9 or later (developed and tested on Python 3.14)
- A webcam (built-in or USB)
- Packages in `requirements.txt`

## 5. Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python colorcount_cv.py
```

To process a file instead of the webcam, pass its path:

```powershell
python colorcount_cv.py sample_inputs/sample_objects.png
python colorcount_cv.py sample_inputs/sample_touching.png
python colorcount_cv.py sample_inputs/sample_video.avi
```

If PowerShell blocks activation, skip it and use `python -m pip ...` directly.
Screenshots are always saved to the `screenshots/` folder next to the script.

**Controls:** `S` saves the processed frame (and the mask view if open) to `screenshots/`, `M` shows/hides the color masks, `Q` quits.

## 6. How to demonstrate it

1. Put a red, green, and blue object on a plain, light background.
2. Start with one object, then add more; show the boxes, labels, and counts updating.
3. Press `M` and explain the masks and contours.
4. Test a difficult case (dim light or touching objects) and explain the limitation.

## 7. Troubleshooting

- **Webcam will not open:** close other apps using the camera, allow camera access in Windows settings, or set `CAMERA_INDEX = 1` in `colorcount_cv.py`.
- **Nothing detected:** improve lighting, use a plain light background, move objects closer. HSV thresholds are starting values; lower `MIN_CONTOUR_AREA` (500) for small objects.
- **Two touching objects count as one:** same-color regions that touch merge into one contour. Keep them apart.

## 8. Limitations and future improvements

Detects color, not object identity. Lighting, shadows, reflections, similar-colored backgrounds, small objects, and touching same-color objects affect results. Ideas: HSV trackbars, detection logs, line-crossing counts, watershed to split touching objects, a trained model for object names.
