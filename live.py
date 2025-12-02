import cv2
import torch
from ultralytics import YOLO
import time
from main import load_config
from src.model import YoloModel


def main():
    # Load YOLOv8n model (replace with your finetuned model if needed)
    config = load_config()
    yolo_model = YoloModel(config)
    model = yolo_model.load_yolo_from_wandb()

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not access webcam.")
        return

    print("📷 Webcam running... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        start = time.time()

        # Run YOLO inference
        results = model(frame, stream=True)

        # Loop through detections
        for r in results:
            for box in r.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                cls = int(box.cls)

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Label text
                label = f"{r.names[cls]} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        end = time.time()
        fps = 1 / (end - start)

        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Show frame
        cv2.imshow("YOLOv8n Webcam", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
