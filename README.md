
# **Image Forgery Detection Using CNN**

## **Overview**
This project aims to detect image forgery using Convolutional Neural Networks (CNN). The model is trained on a dataset of authentic and forged images and uses Grad-CAM visualization for interpretability. The project includes data preparation, model training, evaluation, and visualization of predictions.

---

## **Features**
- **Data Preparation**: Automated data preprocessing and augmentation.
- **CNN Architecture**: Regularized CNN with dropout layers to prevent overfitting.
- **Evaluation Metrics**: Accuracy, precision, recall, confusion matrix, and ROC-AUC score.
- **Explainable AI**: Grad-CAM visualization for understanding model predictions.
- **End-to-End Workflow**: From data preparation to model evaluation and visualization.

---

## **Dataset**
The dataset consists of two categories:
1. **Authentic Images**: Real images without any manipulation.
2. **Forged Images**: Images that have been tampered with or manipulated.

### **Dataset Structure**
The dataset should follow this directory structure:
```
dataset/
├── train/
│   ├── authentic/
│   └── forged/
└── test/
    ├── authentic/
    └── forged/
```

Each folder contains images corresponding to its category.

---

## **Installation**

### **Prerequisites**
Ensure you have the following installed:
- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV

### **Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com//image-forgery-detection.git
   cd image-forgery-detection
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the `dataset/` folder as per the structure mentioned above.

---

## **Usage**

### **Training the Model**
Run the main script to train the model:
```bash
python model_training.py
```
This script will:
1. Preprocess the dataset.
2. Train the CNN model.
3. Evaluate the model on test data.
4. Save the best model as `best_model.h5`.

### **Visualizing Grad-CAM Explanations**
The script also generates Grad-CAM visualizations for a few test samples to help interpret the model's predictions.

---

## **Project Structure**
```
image-forgery-detection/
├── dataset/                # Folder containing train/test data
│   ├── train/
│   │   ├── authentic/
│   │   └── forged/
│   └── test/
│       ├── authentic/
│       └── forged/
├── requirements.txt        # List of dependencies
├── model_training.py       # Main Python script for training and evaluation
├── best_model.h5           # Saved best model weights after training
└── README.md               # Project documentation file
```

---

## **Results**

### **Evaluation Metrics**
After training, the model is evaluated using:
- Accuracy: Measures overall correctness of predictions.
- Precision: Measures correctness for positive predictions (forged images).
- Recall: Measures ability to identify all positive cases.
- Confusion Matrix: Provides insight into true/false positives and negatives.
- ROC-AUC Score: Evaluates the model's ability to distinguish between classes.

### **Grad-CAM Visualization**
Grad-CAM heatmaps overlay on test images to highlight regions that influenced predictions.

---

## **Examples**

### **Confusion Matrix**
1. Experiment with deeper architectures (e.g., ResNet, EfficientNet).
2. Implement multi-class classification for different types of forgery.
3. Use transfer learning to improve performance on small datasets.

---
