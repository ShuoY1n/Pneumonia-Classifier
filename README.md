# Pneumonia Classifier

A deep learning project that uses chest X-ray images to classify whether a patient has pneumonia or is normal using a ResNet-18 model.

## What it does

- Trains a convolutional neural network to classify chest X-ray images
- Distinguishes between normal chest X-rays and those showing pneumonia
- Uses transfer learning with a pre-trained ResNet-18 model
- Provides training, validation, and testing with accuracy metrics
- Saves the trained model for future use

## Motivation

I wanted to explore deep learning for medical image classification and understand how to implement transfer learning with PyTorch for a real-world healthcare application.

## Technologies Used

- **Deep Learning**: PyTorch with torchvision
- **Model Architecture**: ResNet-18 with transfer learning
- **Image Processing**: PIL (Pillow) for image handling
- **Machine Learning**: scikit-learn for evaluation metrics
- **Data Loading**: PyTorch DataLoader for efficient training

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Organize your chest X-ray dataset in folders: `train/NORMAL/`, `train/PNEUMONIA/`, etc.
3. Update the dataset paths in `train_model.py`
4. Run: `python train_model.py`

## Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Credits

Chest X-ray images sourced from [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
