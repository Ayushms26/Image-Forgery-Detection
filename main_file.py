import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kaggle.api.kaggle_api_extended import KaggleApi
import cv2

# Load dataset from kaggle
def setup_kaggle_api(username, key):
    """Securely setup Kaggle API credentials"""
    try:
        possible_locations = [
            os.path.expanduser("~/.kaggle/kaggle.json"),
            os.path.expanduser("~/.config/kaggle/kaggle.json")
        ]
        
        for kaggle_path in possible_locations:
            kaggle_dir = os.path.dirname(kaggle_path)
            try:
                os.makedirs(kaggle_dir, exist_ok=True)
                credentials = {"username": username, "key": key}
                with open(kaggle_path, "w") as f:
                    json.dump(credentials, f)
                os.chmod(kaggle_path, 0o600)
                return True
            except Exception:
                continue
        raise Exception("Could not find a suitable location for kaggle.json")
    except Exception as e:
        print(f"❌ Error setting up Kaggle API: {str(e)}")
        return False

def download_dataset(dataset_name, download_path):
    try:
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.config/kaggle")
        api = KaggleApi()
        api.authenticate()
        print(f"⏳ Downloading dataset: {dataset_name}")
        os.makedirs(download_path, exist_ok=True)
        api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True,
            quiet=False
        )
        print(f"✅ Dataset downloaded to: {download_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

# DATA PREPARATION
def prepare_data(dataset_path, img_size=(256, 256), batch_size=32):
    """Create train/val/test generators from the downloaded dataset"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # For test data, we'll use a separate directory or split
    # Here we're using the same directory but you should have a separate test set
    test_data = test_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_data, val_data, test_data

# GRAD-CAM VISUALIZATION 
def get_gradcam(model, image, layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[0]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (256, 256))
    return heatmap

def visualize_gradcam(model, test_data, num_samples=3):
    """Visualize Grad-CAM for random samples"""
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # Get random batch and select first image
        batch_idx = np.random.randint(0, len(test_data))
        x_batch, y_batch = test_data[batch_idx]
        img = x_batch[0]
        true_label = y_batch[0]
        
        # Get prediction and heatmap
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        heatmap = get_gradcam(model, img)
        
        # Plot original image
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"True: {'Forged' if true_label else 'Authentic'}\nPred: {pred:.2f}")
        plt.axis('off')
        
        # Plot heatmap overlay
        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(img)
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#  MAIN EXECUTION
if __name__ == "__main__":
    # Configuration
    KAGGLE_USERNAME = "ayush07120"  # Replace with your username
    KAGGLE_KEY = "4108d1016606acb0f66c41b0da4769aa"  # Replace with your key
    DATASET_NAME = "labid93/image-forgery-detection"
    DOWNLOAD_PATH = "data/image_forgery"
    
    # Step 1: Download dataset
    if not setup_kaggle_api(KAGGLE_USERNAME, KAGGLE_KEY):
        exit(1)
    
    if not download_dataset(DATASET_NAME, DOWNLOAD_PATH):
        exit(1)
    
    # Step 2: Prepare data generators
    train_data, val_data, test_data = prepare_data(DOWNLOAD_PATH)
    
    # Step 3: Build and train model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_data, epochs=10, validation_data=val_data)
    
    # Step 4: Evaluate model
    loss, acc = model.evaluate(test_data)
    print(f"Test Accuracy: {acc:.2f}")
    
    y_pred = model.predict(test_data)
    y_true = test_data.classes
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    print(classification_report(y_true, y_pred_classes))
    print("ROC-AUC Score:", roc_auc_score(y_true, y_pred))
    
    # Step 5: Visualize Grad-CAM
    visualize_gradcam(model, test_data, num_samples=3)
