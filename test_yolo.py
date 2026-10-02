from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("bottle.jpg")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        print(
            f"Object: {object_name}, "
            f"Confidence: {confidence:.2f}"
        )