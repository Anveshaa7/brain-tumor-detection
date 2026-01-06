#  Brain Tumor Detection System

AI-Powered MRI Analysis for Brain Tumor Detection and Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models](#models)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies](#technologies)
- [Disclaimer](#disclaimer)
- [Contact](#contact)

##  Overview

This project implements an advanced deep learning system for detecting and classifying brain tumors from MRI scans. The system uses transfer learning with three state-of-the-art CNN architectures and provides explainable AI through GradCAM visualization.

**Key Highlights:**
-  **89.71% accuracy** with VGG19
-  **GradCAM visualization** for tumor localization
-  **Web application** with Flask backend
-  **Comprehensive model comparison**
-  **Automated PDF report generation**

##  Features

- **Multi-Model Architecture**: Trained and compared VGG19, ResNet152, and MobileNetV3
- **Transfer Learning**: Leveraged ImageNet pre-trained weights
- **Fine-Tuning**: Achieved 6% accuracy improvement on VGG19
- **Explainable AI**: GradCAM heatmaps show tumor location
- **Web Interface**: User-friendly Flask web application
- **Real-time Predictions**: Upload MRI and get instant results
- **PDF Reports**: Downloadable diagnostic reports

##  Models

### Performance Comparison

| Model | Test Accuracy | Parameters | Training Time |
|-------|--------------|------------|---------------|
| **VGG19** | **89.71%**  | 20.4M | Medium |
| ResNet152 | 71.19% | 60M | High |
| MobileNetV3 | 70.58% | 5M | Low |

### Classification Categories
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

##  Demo

### Web Application Interface
![Web Interface](results/web_app_screenshot.png)


### Model Comparison
![Comparison](results/final_model_comparison.png)

##  Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/AkashGupta-04/brain-tumor-detection.git
cd brain-tumor-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download trained models** (too large for GitHub)
- Download models from [Google Drive Link]
- Place in `models/` directory

##  Usage

### Training Models
```bash
# Train VGG19
jupyter notebook notebooks/VGG19_Training.ipynb

# Train ResNet152
jupyter notebook notebooks/ResNet152_Training.ipynb

# Train MobileNetV3
jupyter notebook notebooks/MobileNetV3_Training.ipynb
```

### Run Web Application
```bash
python app.py
```

Open browser and navigate to: `http://localhost:5000`

### Make Predictions

1. Upload MRI scan image
2. View real-time prediction
3. See GradCAM heatmap
4. Download PDF report

##  Results

### VGG19 Performance
- **Test Accuracy:** 89.71%
- **Precision:** 89.13%
- **Recall:** 88.41%
- **F1-Score:** 88.76%

### Fine-Tuning Impact
- Before: 83.77%
- After: 89.71%
- **Improvement: +5.94%**

### Key Findings
 VGG19 outperformed deeper architectures  
 Fine-tuning significantly improved accuracy  
 Architecture selection matters more than depth  
 GradCAM provides reliable tumor localization  

##  Technologies

- **Deep Learning:** TensorFlow, Keras
- **Web Framework:** Flask
- **Visualization:** Matplotlib, Seaborn
- **PDF Generation:** ReportLab
- **Image Processing:** OpenCV, Pillow
- **Data Analysis:** NumPy, Pandas
- **Model Architectures:** VGG19, ResNet152, MobileNetV3

##  Disclaimer

**IMPORTANT:** This is an educational and research project. It is **NOT intended for clinical use** and has **NOT been approved** by any regulatory authority (FDA, CE, etc.).

- This system is for demonstration purposes only
- NOT a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- No clinical validation has been performed

See [DISCLAIMER.md](docs/DISCLAIMER.md) for complete details.

##  Project Structure
```
brain-tumor-detection/
├── app.py                    # Flask web application
├── templates/
│   └── index.html           # Web interface
├── notebooks/
│   ├── VGG19_Training.ipynb
│   ├── ResNet152_Training.ipynb
│   ├── MobileNetV3_Training.ipynb
│   └── Final_Model_Comparison.ipynb
├── results/
│   ├── final_model_comparison.png
│   └── model_comparison.csv
├── docs/
│   └── DISCLAIMER.md
├── requirements.txt
├── README.md
└── .gitignore
```

##  Future Improvements

- [ ] 3D MRI volume analysis
- [ ] Ensemble methods combining all models
- [ ] Mobile application deployment
- [ ] Real-time inference optimization
- [ ] Integration with PACS systems
- [ ] Multi-modal MRI analysis (T1, T2, FLAIR)
- [ ] Tumor size estimation
- [ ] Clinical validation studies

##  Contact

**Akash Gupta**
- GitHub: https://github.com/AkashGupta-04
- Email: guptaaa2904@gmail.com
- LinkedIn: https://www.linkedin.com/in/akash-gupta-034a5b305/

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training
- Pre-trained models: ImageNet
- Inspiration: Medical AI research community

---

** Star this repository if you find it helpful!**

Made with for advancing AI in healthcare