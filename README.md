# MediAI System – Medical Image Diagnosis Using Explainable AI

## Overview

MediAI System is a full-stack medical image analysis web application designed to assist in the automated detection of brain tumors and retinal diseases from medical images. The system leverages deep learning models based on ResNet-18 and integrates Grad-CAM to provide explainable visual interpretations of model predictions.

The application supports:
- Brain tumor detection from MRI scans
- Retinal disease detection from OCT/Fundus images

MediAI System is implemented using Flask for the backend, PyTorch for deep learning training and inference, and standard web technologies (HTML, CSS, JavaScript) for the frontend interface.

---

## Purpose and Use Case

The primary purpose of MediAI System is to demonstrate the application of artificial intelligence and explainable deep learning techniques in medical image analysis. The system is suitable for:

- Academic learning and teaching
- Demonstration of Explainable AI (XAI) in healthcare
- Medical AI research prototypes

This project is intended strictly for educational and research purposes and is not suitable for clinical diagnosis.

---

## Key Features

- Web-based interface for medical image upload
- Deep learning inference using trained ResNet-18 models
- Grad-CAM heatmap visualization for explainability
- Separate pipelines for brain MRI and retinal image analysis
- Modular training and inference scripts
- Clean and responsive frontend design

---

## Project Structure


```
MediAI-System/
│
├── app.py
├── brain.py
├── rectina.py
├── train_brain.py
├── train_rectina.py
│
├── models/
│   ├── brain_tumor_resnet18.pth
│   └── octmnist_resnet18.pth
│
├── templates/
│   ├── home.html
│   ├── brain_index.html
│   └── rectina_index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── screenshots/
│   ├── home.png
|   ├── brain-home.png
|   ├── rectina-home.png
|   ├── brain-uploaded.png	
|   ├── brain-uploaded.png	
│   ├── brain-result.png
│   └── rectina-result.png
│
├── requirements.txt 
└── README.md
```

---

## File and Directory Description

### Backend Files

- `app.py`  
  Main Flask application entry point. Manages routing between home, brain tumor detection, and retina disease detection modules.

- `brain.py`  
  Implements brain tumor inference logic including model loading, preprocessing, prediction, and Grad-CAM visualization.

- `rectina.py`  
  Implements retina disease inference logic using MedMNIST-trained models with Grad-CAM visualization.

- `train_brain.py`  
  Training script for the brain tumor detection model using the Brain Tumor MRI Dataset.

- `train_rectina.py`  
  Training script for the retina disease detection model using the MedMNIST OCTMNIST dataset.

---

### Models Directory

- `models/`  
  Contains trained model weights required for inference:
  - `brain_tumor_resnet18.pth`
  - `octmnist_resnet18.pth`

The `models` directory must exist before running inference or training scripts.

---

### Frontend Files

- `templates/`  
  HTML templates rendered by Flask:
  - `home.html` – Application landing page
  - `brain_index.html` – Brain tumor analysis page
  - `rectina_index.html` – Retina disease analysis page

- `static/`  
  Static assets for the frontend:
  - `css/style.css` – Styling and layout
  - `js/main.js` – Client-side interactivity

---

### Screenshots

- `screenshots/`  
  Contains UI screenshots used for documentation:
  - `home.png`
  - `brain-result.png`
  - `rectina-result.png`

---

## Dataset Information

### Brain Tumor Detection
- Dataset: Brain Tumor MRI Dataset
- Source: Kaggle
- Usage: Used to train the brain tumor classification model
- Note: The dataset is not included in this repository due to size and licensing restrictions

### Retina Disease Detection
- Dataset: MedMNIST (OCTMNIST)
- Source: MedMNIST Library
- Usage: Automatically downloaded and loaded using the `medmnist` Python package

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/MediAI-System.git
cd MediAI-System
````

---

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3: Prepare the Models Directory

Before training or running inference, ensure the `models` directory exists:

```bash
mkdir models
```

Place the trained `.pth` files inside the `models/` directory if not training from scratch.

---

## Running the Application

Start the Flask server using:

```bash
python app.py
```

Open the application in a web browser:

```
http://127.0.0.1:5000/
```

---

## Training the Models 

### Train Brain Tumor Detection Model

```bash
python train_brain.py
```

Ensure the Brain Tumor MRI Dataset is available locally and the dataset path matches the configuration in the script.

---

### Train Retina Disease Detection Model

```bash
python train_rectina.py
```

The MedMNIST dataset will be downloaded automatically during training.

---

## Common Issues and Troubleshooting

### ModuleNotFoundError

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

---

### FileNotFoundError for Model Files

Verify that the `models/` directory exists and contains the required `.pth` files.

---

### CUDA Not Available

The application automatically falls back to CPU execution if a GPU is not available. No configuration changes are required.

---

### Dataset Path Errors During Training

Confirm that the dataset directory path in `train_brain.py` matches the local dataset location.

---

## Screenshots Reference

The following screenshots demonstrate the complete application workflow:

Home page interface and navigation (`screenshots/home.png`)

Brain tumor detection upload interface (`screenshots/brain-home.png`)

Retina disease detection upload interface (`screenshots/rectina-home.png`)

Brain MRI image uploaded and previewed before analysis (`screenshots/brain-uploaded.png`)

Brain tumor detection results with Grad-CAM visualization (`screenshots/brain-result.png`)

Retina disease detection results with Grad-CAM visualization (`screenshots/rectina-result.png`)


---

## Disclaimer

This project is intended for educational and research purposes only. It is not certified for clinical use and must not be used for real medical diagnosis or treatment decisions.

---

## Author

Akshay A C




