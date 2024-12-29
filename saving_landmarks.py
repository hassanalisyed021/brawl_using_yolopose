# File: save_landmarks.py
import cv2
from ultralytics import YOLO
import numpy as np
import os

def save_landmarks(output_file="landmarks.txt"):
    # Load YOLOv8 model
    model = YOLO("yolov8n-pose.pt")  # Ensure you have the correct YOLOv8-Pose model

    cap = cv2.VideoCapture('trainthis.mp4')  # Use video file or webcam
    landmarks_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose estimation
        results = model(frame, save=False, conf=0.5, stream=True)

        for result in results:
            if result.keypoints is not None:  # Check if keypoints are detected
                keypoints = result.keypoints  # Get Keypoints object

                # Extract (x, y) coordinates and confidence values
                xy_coords = keypoints.xy.numpy()  # Convert to numpy (Shape: [num_detections, num_keypoints, 2])
                conf_values = keypoints.conf.numpy()  # Convert to numpy (Shape: [num_detections, num_keypoints])

                # Ensure dimensions match: reshape conf_values to match xy_coords
                if len(xy_coords.shape) == 3 and len(conf_values.shape) == 2:
                    conf_values = conf_values[..., np.newaxis]  # Add an axis for concatenation (Shape: [num_detections, num_keypoints, 1])
                    keypoints_combined = np.concatenate([xy_coords, conf_values], axis=2)  # Combine along the last dimension

                    # Flatten the array for saving
                    for detection in keypoints_combined:  # Iterate over detections
                        keypoints_flat = detection.flatten()  # Flatten each detection
                        landmarks_list.append(keypoints_flat.tolist())

                        # Save to file
                        with open(output_file, "a") as f:
                            f.write(",".join(map(str, keypoints_flat)) + "\n")

                # Visualize
                annotated_frame = result.plot()
                cv2.imshow("Pose Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_landmarks()
