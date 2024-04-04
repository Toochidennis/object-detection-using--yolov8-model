from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

#results = model.train(source=0, show=True, conf=0.4, save=True)

cap = cv2.VideoCapture(0)


# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Loop through video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    detection_results = model(frame)

    if detection_results:
        for result in detection_results:
            if result.boxes:
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0]

                # Convert tensor values to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                name = result.names[box.cls[0].item()]
                confidence = box.conf[0].item()

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the name and confidence level at the top of the bounding box
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with bounding boxes and text
    cv2.imshow("Object detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
