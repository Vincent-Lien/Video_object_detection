import argparse
import cv2
import os

def video_to_frames(input_video, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_video)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--input_video", type=str, help="Path to the input video file.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder to save frames.")
    args = parser.parse_args()

    video_to_frames(args.input_video, args.output_folder)