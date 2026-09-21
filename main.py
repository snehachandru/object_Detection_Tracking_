from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot access webcam")
        break

    # Object detection + tracking
    results = model.track(frame, persist=True)

    # Draw detected objects and tracking IDs
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("Object Detection and Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()