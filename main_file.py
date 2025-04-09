import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import cv2

# 1. High-Resolution Image Loader (No Downscaling)
def load_highres_images(folder, label, target_size=None):
    images, labels = [], []
    for filename in os.listdir(folder)[:500]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            if target_size:  # Optional resizing (not used here)
                img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# 2. Load CASIA v2 and CoMoFoD - Splicing + Copy-Move Forgeries
def load_datasets(casia_path, comofod_path):
    casia_auth, y_casia_auth = load_highres_images(f'{casia_path}/Au', 0)
    casia_forged, y_casia_forged = load_highres_images(f'{casia_path}/Tp', 1)
    
    comofod_auth, y_como_auth = load_highres_images(f'{comofod_path}/original', 0)
    comofod_forged, y_como_forged = load_highres_images(f'{comofod_path}/forged', 1)
    
    # Combine
    X = np.concatenate((casia_auth, casia_forged, comofod_auth, comofod_forged))
    y = np.concatenate((y_casia_auth, y_casia_forged, y_como_auth, y_como_forged))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. EfficientNet-Based CNN without ELA
def build_high_acc_model(input_shape):
    base_model = applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Use pretrained features
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 4. Evaluation: Accuracy, F1, ROC-AUC
def full_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels))

    f1 = f1_score(y_test, y_pred_labels)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"\nF1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return f1, roc_auc

# 5. Main Execution Pipeline
def main():
    CASIA_PATH = 'path/to/CASIAv2'
    COMOFOD_PATH = 'path/to/CoMoFoD'

    print("\n[INFO] Loading high-resolution datasets...")
    X_train, X_test, y_train, y_test = load_datasets(CASIA_PATH, COMOFOD_PATH)

    input_shape = X_train[0].shape
    print(f"[INFO] Training on image shape: {input_shape}")

    print("\n[INFO] Building model architecture...")
    model = build_high_acc_model(input_shape)

    print("\n[INFO] Starting model training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=16
    )

    print("\n[INFO] Evaluating performance...")
    f1, roc_auc = full_evaluation(model, X_test, y_test)

    if f1 >= 0.94 and roc_auc >= 0.96:
        model.save("forgery_detector.h5")
        print("\n[SUCCESS] Model saved successfully with target performance achieved.")
    else:
        print("\n[WARNING] Model did not meet all target metrics.")

if __name__ == '__main__':
    main()
