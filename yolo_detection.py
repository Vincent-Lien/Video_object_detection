from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os

# Load a pretrained YOLO11n model
model_weight_path = "yolo11x.pt"
model = YOLO(model_weight_path)

# Define path to video file
source = "person_cats_V3.mp4"

# Open the video file
cap = cv2.VideoCapture(source)

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Create the output directory if it doesn't exist
output_dir = 'video'
os.makedirs(output_dir, exist_ok=True)

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video/output_yolo.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Run inference on the source

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_count = 0
    cat_count = 0
    
    for result in results:        
        # Draw bounding boxes
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.1:
                continue
            
            if box.cls in [0, 15]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls)]
                
                if box.cls == 0:  # Person
                    color = (0, 0, 255) # Red
                    person_count += 1
                elif box.cls == 15:  # Cat
                    color = (255, 0, 255)   # Magenta
                    cat_count += 1
            
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Add text to the top-left corner of the frame in red and bold, split into three lines
    cv2.putText(frame, '313581009', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Person: {person_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Cats: {cat_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)

    # Open CSV file in append mode
    with open('frame_log/yolo_detection.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(['frame', 'person', 'cat'])
        # Write the frame number, person count, and cat count
        csvwriter.writerow([int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, person_count, cat_count])

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()