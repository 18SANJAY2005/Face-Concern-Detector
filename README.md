# Face Concern Detector

A deep learning model that detects facial concerns (acne, dark circles, redness, wrinkles) from images using PyTorch and MTCNN face detection. The model uses Grad-CAM to generate heat maps that highlight areas of detected concerns, making it useful for both analysis and visualization.

## Features

- 🔍 **Face Detection & Alignment**: Automatically detects and aligns faces using MTCNN
- 🧠 **Multi-Label Classification**: Analyzes four common facial concerns:
  - Acne
  - Dark Circles
  - Redness
  - Wrinkles
- 📊 **Confidence Scores**: Provides confidence percentages for each detected concern
- 🗺️ **Visualization**: Generates heat maps using Grad-CAM to highlight areas of concern
- 📱 **Easy Integration**: Includes both Jupyter notebook demo and importable Python modules
- 🔄 **Batch Processing**: Support for both file and in-memory image processing

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=10.0.0
mtcnn>=0.1.1
jupyter>=1.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pandas>=2.0.0
```

## Project Structure

```
├── face_scanner_demo.ipynb    # Demo notebook
├── requirements.txt           # Project dependencies
├── models/
│   └── best_model.pth        # Trained model weights
├── outputs/                   # Directory for output visualizations
├── sample_images/            # Sample images for testing
└── src/
    ├── __init__.py
    ├── gradcam.py           # GradCAM implementation
    ├── inference.py         # Inference pipeline
    ├── model.py            # Model architecture
    ├── preprocessing.py    # Image preprocessing
    └── visualization.py    # Visualization utilities
```

## Getting Started

1. Clone the repository
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the demo notebook:
   ```bash
   jupyter notebook face_scanner_demo.ipynb
   ```

## Usage

### Basic Usage
```python
from src.model import load_model
from src.inference import predict

# Load the model
model = load_model('models/best_model.pth')

# Analyze an image
scores, heatmaps, face_image = predict(
    model=model,
    image_path='path/to/your/image.jpg',
    generate_heatmaps=True
)

# Print the results
print(scores)
```

### Using with Byte Data
```python
from src.inference import predict_from_bytes

# For processing image data from memory
scores, overlay_path = predict_from_bytes(
    model=model,
    image_bytes=your_image_bytes
)
```

## Model Architecture

The Face Concern Detector uses a modified ResNet-18 architecture with:
- Pre-trained ImageNet weights
- Custom classifier head with dropout layers
- Sigmoid activation for multi-label classification
- Output predictions for 4 different facial concerns

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- MTCNN implementation from [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
- Grad-CAM visualization adapted from the [official paper implementation](https://github.com/jacobgil/pytorch-grad-cam)
- ResNet architecture from torchvision