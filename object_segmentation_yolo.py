from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLO11n model
model = YOLO("yolo11x-seg.pt")

# Define path to video file
source = "person_cats_V3.mp4"

# Open the video file
cap = cv2.VideoCapture(source)

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_seg_conf_0.4.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Run inference on the source

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_count = 0
    cat_count = 0
    
    for result in results:
        # Draw segmentation masks
        if result.masks is not None:
            for seg, cls, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
                if conf < 0.4:
                    continue
                if cls == 0:
                    color = (0, 0, 255)  # Red
                    person_count += 1
                elif cls == 15:
                    color = (255, 0, 255)  # Purple
                    cat_count += 1
                else:
                    continue

                overlay = frame.copy()
                points = np.array(seg, dtype=np.int32)
                cv2.fillPoly(overlay, [points], color)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Draw bounding boxes
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < 0.4:
                continue
            
            if box.cls in [0, 15]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls)]
                
                if box.cls == 0:  # Person
                    color = (0, 0, 255)
                elif box.cls == 15:  # Cat
                    color = (255, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Add text to the top-left corner of the frame in red and bold, split into three lines
    cv2.putText(frame, '313581009', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Persons: {person_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Cats: {cat_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()