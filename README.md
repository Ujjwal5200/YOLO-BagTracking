# YOLO Box Tracking on Conveyor Belt

A computer vision project that utilizes YOLOv11 for real-time object detection and tracking of boxes/products on a conveyor belt system. The system detects, tracks, and counts incoming and outgoing items with high accuracy.

## 🚀 Features

- **Real-time Object Detection**: Uses YOLOv11 for accurate detection of boxes/products
- **Object Tracking**: Implements robust tracking across video frames
- **Directional Counting**: Counts items moving in both directions (incoming/outgoing)
- **Class-wise Counting**: Separate counts for different product types
- **Visual Annotations**: Real-time overlay of bounding boxes, tracking IDs, and counts
- **Video Processing**: Processes input video and generates annotated output video

## 📊 Dataset

The project uses a custom dataset from Roboflow containing:
- **388 images** of boxes on conveyor belts
- **2 classes**: 'product 1' and 'product 2'
- **Preprocessing**: Auto-orientation, resize to 640x640
- **Augmentations**: Horizontal/vertical flips, rotations, cropping, exposure adjustments, blur

Dataset source: [Roboflow Universe - Box on Convey](https://universe.roboflow.com/insightemporium-umywk/box-on-convey-tamvr)

## 🏗️ Model Training

### Training Configuration
- **Model**: YOLOv11n (pre-trained)
- **Epochs**: 100
- **Image Size**: 640x640
- **Optimizer**: Default YOLO optimizer with learning rate scheduling

### Training Results
The model achieved excellent performance:
- **Final mAP50-95**: 0.950
- **Final mAP50**: 0.995
- **Precision**: 0.994
- **Recall**: 0.990

Training metrics improved steadily over 100 epochs, with significant gains in the first 20 epochs and continued refinement thereafter.

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training/inference)

### Dependencies
```bash
pip install ultralytics opencv-python supervision
```

### Clone and Setup
```bash
git clone <repository-url>
cd cement_YOLO
```

## 📁 Project Structure

```
cement_YOLO/
├── main.py                 # Main inference script
├── main.ipynb             # Jupyter notebook with training and inference
├── dataset/               # Dataset directory
│   ├── data.yaml         # Dataset configuration
│   ├── train/            # Training images and labels
│   ├── valid/            # Validation images and labels
│   └── README.*          # Dataset documentation
├── runs/                  # Training outputs
│   └── detect/
│       └── train5/       # Best trained model
├── yolo11n.pt            # Base YOLOv11 model
├── yolo11m.pt            # Alternative model (not used)
├── video1.mp4            # Input video
├── output_video.avi      # Processed output video
└── README.md             # This file
```

## 🚀 Usage

### Training (Optional - Model already trained)
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolo11n.pt')

# Train on custom dataset
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640
)
```

### Inference and Tracking
Run the main script for real-time detection and counting:
```bash
python main.py
```

This will:
1. Load the trained model (`runs/detect/train5/weights/best.pt`)
2. Process `video1.mp4`
3. Generate `output_video.avi` with annotations
4. Display real-time counts for incoming/outgoing items

### Key Parameters
- **Line Positions**: Red line at x=150, Blue line at x=200 (adjustable)
- **Counting Logic**: Items crossing lines in specific directions are counted
- **Output**: Annotated video with bounding boxes, tracking IDs, and counts

## 📈 Results and Performance

### Detection Performance
- High accuracy detection of boxes on moving conveyor
- Robust tracking through occlusions and varying lighting
- Real-time processing capabilities

### Counting Accuracy
- Accurate directional counting (incoming vs outgoing)
- Class-wise breakdown of counts
- Minimal false positives/negatives

### Sample Output
The system provides:
- Total incoming/outgoing counts
- Per-class counts (product 1, product 2)
- Visual tracking with unique IDs
- Real-time overlay on video feed

## 🔍 Technical Details

### Tracking Algorithm
- Uses YOLO's built-in tracking with `persist=True`
- Maintains track IDs across frames
- Handles object entry/exit from frame

### Counting Mechanism
- Vertical line-based counting
- Direction determination based on line crossing order
- Prevents double-counting with ID tracking

### Video Processing
- Maintains original video properties (fps, resolution)
- Outputs in XVID AVI format
- Real-time display with OpenCV

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses the CC BY 4.0 license for the dataset and MIT license for the code.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLOv11 implementation
- [Roboflow](https://roboflow.com) for the dataset platform
- [OpenCV](https://opencv.org) for computer vision utilities
- [Supervision](https://github.com/roboflow/supervision) for annotation tools

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
