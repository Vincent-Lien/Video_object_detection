from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import csv

### Florence Model ###
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


### YOLO ###
yolo = YOLO("yolo11x.pt")


### Video Processing ###

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
out = cv2.VideoWriter('video/output_florence_with_yolo.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))


bbox_task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Florence detection
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox_results = run_example(input_image, bbox_task_prompt, text_input="cats and humans")
    for i, label in enumerate(bbox_results[bbox_task_prompt]['labels']):
        if label == "cats":
            bbox_results[bbox_task_prompt]['labels'][i] = "cat"
        elif label == "cat":
            continue
        else:
            bbox_results[bbox_task_prompt]['labels'][i] = "person"

    # YOLO detection
    yolo_results = yolo(frame)

    person_count = 0
    cat_count = 0
    # Draw YOLO detection boxes
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            if label == "cat":
                cat_count += 1
                color = (255, 0, 255)  # Purple color for cats
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            elif label == "person":
                person_count += 1
                color = (0, 0, 255)  # Red color for persons
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Check Florence results and add missing boxes
    for label, box in zip(bbox_results[bbox_task_prompt]['labels'], bbox_results[bbox_task_prompt]['bboxes']):
        x1, y1, x2, y2 = map(int, box)
        if label == "cat":
            color = (255, 0, 255)  # Purple color for cats
        elif label == "person":
            color = (0, 0, 255)  # Red color for persons
        # Check if the box is already drawn by YOLO
        if not any((x1 <= int(b.xyxy[0][0]) <= x2 and y1 <= int(b.xyxy[0][1]) <= y2) or 
               (abs(x1 - int(b.xyxy[0][0])) < 20 and abs(y1 - int(b.xyxy[0][1])) < 20 and 
                abs(x2 - int(b.xyxy[0][2])) < 20 and abs(y2 - int(b.xyxy[0][3])) < 20) 
                for result in yolo_results for b in result.boxes if result.names[int(b.cls[0])] in ["cat", "person"]):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # Increment the count for cats or persons
            if label == "cat":
                cat_count += 1
            elif label == "person":
                person_count += 1

    # Add text to the top-left corner of the frame in red and bold, split into three lines
    cv2.putText(frame, f'Person: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Cats: {cat_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)

    # Open CSV file in append mode
    with open('frame_log/yolo_florence_detection.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(['frame', 'person', 'cat'])
        # Write the frame number, person count, and cat count
        csvwriter.writerow([int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, person_count, cat_count])

cap.release()
out.release()
cv2.destroyAllWindows()