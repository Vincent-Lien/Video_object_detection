# Video_object_detection
This project focuses on human and cat detection in video footage, utilizing three different detection methods:
1. YOLO v11
2. Florence-2
3. YOLO v11 + Florence-2 (combined approach)

## Setup & Installation
1. Clone the repository:
```bash
git clone https://github.com/Vincent-Lien/Video_object_detection.git
cd Video_object_detection
```

2. Set up Python environment (Python 3.10):
```bash
conda create -n video_detect python=3.10
conda activate video_detect
pip install -r requirements.txt
```

## Project Components
### Project Files
- `person_cats_V3.mp4`: Input video for processing
- `person_cats_baseline.mp4`: Baseline video for comparison
- `yolo_detection.py`: YOLO implementation
- `florence_detection.py`: Florence-2 implementation
- `yolo_florence_detection.py`: Combined detection
- `video2frame.py`: Frame extraction tool
- `evaluation.ipynb`: Analysis and visualization

### Output Directories
- `/video/`: Processed videos with detections
- `/frame_log/`: Detection count logs (CSV)
- `/video_frames/`: Extracted frames

## Demo Videos
Please check out the demo videos on this [YouTube playlist](https://www.youtube.com/watch?v=GZvXXO5-d6g&list=PL5CSYCnSztknhQjGjRP9OFIskxG2mWMCM).

Alternatively, you can find the demo videos in the [video directory](./video/).

## Usage
### YOLO Detection
```bash
python yolo_detection.py
```

### Florence Detection
```bash
python florence_detection.py
```

### Combined Detection
```bash
python yolo_florence_detection.py
```

### Frame Extraction
```bash
python video2frame.py --input_video path/to/video.mp4 --output_folder path/to/output/
```
Parameters:
- `--input_video`: Input video path
- `--output_folder`: Output directory for frames

### Evaluation
Open `evaluation.ipynb` in Jupyter Notebook to analyze detection results and view visualizations.