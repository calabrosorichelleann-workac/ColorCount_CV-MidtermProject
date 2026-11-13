"""
ColorCount CV
Real-Time Color-Based Object Detection and Counting System

Usage:
    python colorcount_cv.py                       -> live webcam
    python colorcount_cv.py sample_inputs/sample_objects.png   -> static image
    python colorcount_cv.py sample_inputs/sample_video.avi     -> recorded video

Controls:
    Q - Quit the application
    S - Save a screenshot of the detection results
        (also saves the color masks if the mask window is open)
    M - Show / hide the color mask window

Features:
    - Detects RED, GREEN, and BLUE objects using HSV color segmentation.
    - Uses Gaussian blur and morphological operations to reduce noise.
    - Detects object boundaries using contour detection.
    - Displays bounding boxes, color labels, and object counts.
    - Shows the binary color masks used for detection (press M).
    - Works on a live webcam feed, a static image, or a recorded video.
    - Saves processed screenshots in the screenshots folder.

Note:
    This is classical computer vision using OpenCV,
    not AI-based object recognition.
"""

import sys
import cv2
import numpy as np
import os
from datetime import datetime


# ==========================================
# 1. SETTINGS
# ==========================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Ignore tiny contours/noise below this area.
MIN_CONTOUR_AREA = 500

# File extensions treated as static images (anything else = video file).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Screenshots are always saved next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(SCRIPT_DIR, "screenshots")

# Window names.
MAIN_WINDOW = "ColorCount CV - Live Detection"
MASK_WINDOW = "ColorCount CV - Color Masks"

# HSV color ranges. OpenCV hue range is 0-179.
# Red uses two ranges because it wraps around the HSV scale.

COLOR_RANGES = {
    "RED": [
        (np.array([0, 100, 70]), np.array([10, 255, 255])),
        (np.array([170, 100, 70]), np.array([179, 255, 255])),
    ],

    "GREEN": [
        (np.array([35, 70, 50]), np.array([85, 255, 255])),
    ],

    "BLUE": [
        (np.array([90, 70, 50]), np.array([130, 255, 255])),
    ],
}

# Bounding box colors in BGR format.

BOX_COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
}

# Kernel for morphological operations.

KERNEL = np.ones((5, 5), np.uint8)


# ==========================================
# 2. BUILD COLOR MASK
# ==========================================

def build_color_mask(hsv_frame, ranges):
    """
    Combine HSV masks for a specific color
    and reduce small image noise.
    """

    combined_mask = np.zeros(
        hsv_frame.shape[:2],
        dtype=np.uint8
    )

    for lower, upper in ranges:

        mask = cv2.inRange(
            hsv_frame,
            lower,
            upper
        )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            mask
        )

    # Opening removes small white specks.
    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    # Closing fills small gaps.
    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    return combined_mask


# ==========================================
# 3. DETECT AND COUNT OBJECTS
# ==========================================

def detect_objects(frame, hsv_frame):
    """
    Detect colored regions, draw bounding boxes,
    display labels, and return object counts.
    """

    counts = {
        "RED": 0,
        "GREEN": 0,
        "BLUE": 0
    }

    masks = {}

    for color_name, ranges in COLOR_RANGES.items():

        # Create a binary mask for the selected color.
        mask = build_color_mask(
            hsv_frame,
            ranges
        )

        masks[color_name] = mask

        # Find outer boundaries of detected regions.
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore small regions considered noise.
            if area < MIN_CONTOUR_AREA:
                continue

            # Get the bounding rectangle.
            x, y, w, h = cv2.boundingRect(contour)

            counts[color_name] += 1

            box_color = BOX_COLORS[color_name]

            # Draw bounding box.
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # Display color label and contour area.
            cv2.putText(
                frame,
                f"{color_name} ({int(area)} px)",
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA
            )

    return frame, counts, masks


# ==========================================
# 4. DRAW DASHBOARD
# ==========================================

def draw_dashboard(frame, counts):
    """
    Display the application title,
    individual color counts, total count,
    and keyboard instructions.
    """

    total = sum(counts.values())

    # Create a dark dashboard panel.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (250, 125),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame
    )

    # Application title.
    cv2.putText(
        frame,
        "ColorCount CV",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Red count.
    cv2.putText(
        frame,
        f"RED: {counts['RED']}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        BOX_COLORS["RED"],
        2
    )

    # Green count.
    cv2.putText(
        frame,
        f"GREEN: {counts['GREEN']}",
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        BOX_COLORS["GREEN"],
        2
    )

    # Blue count.
    cv2.putText(
        frame,
        f"BLUE: {counts['BLUE']}",
        (18, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        BOX_COLORS["BLUE"],
        2
    )

    # Total detected objects.
    cv2.putText(
        frame,
        f"TOTAL OBJECTS: {total}",
        (frame.shape[1] - 250, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Keyboard instructions.
    cv2.putText(
        frame,
        "S: Save | M: Masks | Q: Quit",
        (frame.shape[1] - 280, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return frame


# ==========================================
# 5. COLOR MASK VIEW
# ==========================================

def build_mask_view(masks, shape):
    """
    Combine the RED, GREEN, and BLUE binary masks into one
    color image (black background) so the segmentation
    result can be shown on screen.
    """

    mask_view = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    for color_name, mask in masks.items():
        mask_view[mask > 0] = BOX_COLORS[color_name]

    cv2.putText(
        mask_view,
        "Segmentation masks (red / green / blue)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return mask_view


# ==========================================
# 6. PROCESS ONE FRAME
# ==========================================

def process_frame(frame):
    """
    Run the full pipeline on one BGR frame:
    resize -> blur -> HSV -> masks -> contours -> overlays.
    Returns the processed frame, the counts, and the masks.
    """

    # IMAGE OPERATIONS
    # Resize the image.
    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    # Apply Gaussian blur.
    blurred = cv2.GaussianBlur(
        frame,
        (5, 5),
        0
    )

    # Convert BGR to HSV.
    hsv_frame = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # IMAGE SEGMENTATION AND ANALYSIS
    processed_frame, counts, masks = detect_objects(
        frame.copy(),
        hsv_frame
    )

    # Draw the dashboard.
    processed_frame = draw_dashboard(
        processed_frame,
        counts
    )

    return processed_frame, counts, masks


# ==========================================
# 7. SAVE SCREENSHOT
# ==========================================

def save_screenshot(frame, mask_view=None):
    """
    Save the processed frame (and the mask view, if given)
    inside the screenshots folder next to this script.
    """

    # Create the folder if it does not exist.
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Generate a unique filename.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = os.path.join(
        SCREENSHOT_DIR,
        f"detection_{timestamp}.png"
    )

    # Save the processed image.
    success = cv2.imwrite(
        filename,
        frame
    )

    if success:
        print(f"Screenshot saved successfully: {filename}")
    else:
        print("ERROR: Could not save screenshot.")

    # Save the mask view too when it is open.
    if mask_view is not None:

        mask_filename = os.path.join(
            SCREENSHOT_DIR,
            f"masks_{timestamp}.png"
        )

        if cv2.imwrite(mask_filename, mask_view):
            print(f"Mask view saved successfully: {mask_filename}")


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

def open_source(source):
    """
    Decide what to process.
    Returns (camera, image): camera is a VideoCapture for the
    webcam or a video file, image is a loaded static image.
    Exactly one of them is not None; both None means failure.
    """

    # No argument: live webcam.
    if source is None:

        camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():

            print("ERROR: Could not open the webcam.")
            print("Try closing other apps using the camera.")
            print("You may also change CAMERA_INDEX.")

            return None, None

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        return camera, None

    # A file path was given.
    if not os.path.isfile(source):

        print(f"ERROR: File not found: {source}")

        return None, None

    # Static image.
    if source.lower().endswith(IMAGE_EXTENSIONS):

        image = cv2.imread(source)

        if image is None:

            print(f"ERROR: Could not read the image: {source}")

            return None, None

        return None, image

    # Recorded video.
    video = cv2.VideoCapture(source)

    if not video.isOpened():

        print(f"ERROR: Could not open the video: {source}")

        return None, None

    return video, None


def main():

    # Optional first argument: path to an image or video file.
    source = sys.argv[1] if len(sys.argv) > 1 else None
    is_webcam = source is None

    camera, static_image = open_source(source)

    if camera is None and static_image is None:
        return

    print("====================================")
    print("       ColorCount CV Started")
    print("====================================")
    print("Input: " + ("live webcam" if is_webcam else source))
    print("Press S to save a screenshot.")
    print("Press M to show/hide the color masks.")
    print("Press Q to quit.")
    print("====================================")

    show_masks = False

    try:

        # Continue processing frames.
        while True:

            # Get the next frame.
            if static_image is not None:

                frame = static_image.copy()

            else:

                success, frame = camera.read()

                if not success:

                    # A video file simply ends; a webcam failing is an error.
                    if is_webcam:
                        print("ERROR: Could not read webcam frame.")
                    else:
                        print("End of video.")

                    break

                # Mirror the webcam view only (not files).
                if is_webcam:
                    frame = cv2.flip(frame, 1)

            processed_frame, counts, masks = process_frame(frame)

            # Display processed frame.
            cv2.imshow(
                MAIN_WINDOW,
                processed_frame
            )

            # Display the segmentation masks when enabled.
            mask_view = None

            if show_masks:

                mask_view = build_mask_view(
                    masks,
                    processed_frame.shape
                )

                cv2.imshow(MASK_WINDOW, mask_view)

            # KEYBOARD CONTROLS
            # Webcam: poll quickly. Image/video files: slower polling
            # so a static image does not use 100% CPU.
            key = cv2.waitKey(1 if is_webcam else 30) & 0xFF

            # Press Q to quit.
            if key == ord("q"):
                print("Closing ColorCount CV...")
                break

            # Press S to save a screenshot.
            elif key == ord("s"):
                save_screenshot(processed_frame, mask_view)

            # Press M to toggle the mask window.
            elif key == ord("m"):

                show_masks = not show_masks

                if not show_masks:
                    cv2.destroyWindow(MASK_WINDOW)

    finally:

        # Release the video source after the loop ends.
        if camera is not None:
            camera.release()

        # Close all OpenCV windows.
        cv2.destroyAllWindows()

        print("ColorCount CV closed.")


# ==========================================
# 9. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    main()
