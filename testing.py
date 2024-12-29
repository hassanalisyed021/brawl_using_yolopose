# File: detect_brawl.py
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

def real_time_detection(model_path="lstm_model.h5", seq_length=30):
    # Load YOLOv8-Pose model and LSTM model
    pose_model = YOLO("yolov8n-pose.pt")
    lstm_model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture('trainthis.mp4')  # Webcam input
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose estimation
        results = pose_model(frame, save=False, conf=0.5, stream=True)

        for result in results:
            # Check if keypoints are detected
            if result.keypoints is not None and result.keypoints.xy is not None and result.keypoints.conf is not None:
                keypoints = result.keypoints

                # Extract (x, y) coordinates and confidence values
                xy_coords = keypoints.xy[0].numpy()  # Shape: (num_keypoints, 2)
                conf_values = keypoints.conf[0].numpy()  # Shape: (num_keypoints,)

                # Combine x, y, and confidence into a single array
                keypoints_combined = np.hstack([xy_coords, conf_values.reshape(-1, 1)])  # Shape: (num_keypoints, 3)
                keypoints_flat = keypoints_combined.flatten()  # Flatten to 1D array

                sequence.append(keypoints_flat)

                if len(sequence) == seq_length:
                    sequence_array = np.expand_dims(sequence, axis=0)
                    prediction = lstm_model.predict(sequence_array)[0][0]
                    sequence.pop(0)  # Maintain sequence length

                    # Display prediction
                    label = "Fight" if prediction > 0.5 else "Not Fight"
                    cv2.putText(frame, f"Prediction: {label}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                annotated_frame = result.plot()
                cv2.imshow("Brawl Detection", annotated_frame)
            else:
                # If no keypoints, continue
                cv2.imshow("Brawl Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
