import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Android phone IP
PHONE_IP = "192.0.0.4"

# Phone video stream
url = f"http://{PHONE_IP}:8080/video"

# Open phone camera stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("ERROR: Could not connect to phone camera")
    exit()

print("Phone camera + YOLO started!")
print("Press Q to quit.")

while True:

    # Read frame from phone
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Phone Camera + YOLO", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()