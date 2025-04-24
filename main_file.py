import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import io

# ======================
# CNN Model Architecture
# ======================

def create_forgery_detection_model(input_shape=(512, 512, 3)):
    model = models.Sequential([
        # Feature Extraction Backbone
        layers.Conv2D(32, (7,7), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Attention Mechanism
        layers.Conv2D(1, (1,1), activation='sigmoid'),
        layers.Multiply(),
        
        # Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.AUC(name='roc_auc')])
    return model

# ====================
# Grad-CAM Implementation
# ====================

def generate_gradcam(model, img_array, layer_name='conv2d_2'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return heatmap

# ====================
# Flask API
# ====================

app = Flask(__name__)
model = create_forgery_detection_model()
model.load_weights('forgery_detection_model.h5')  # Pretrained weights

@app.route('/detect', methods=['POST'])
def detect_forgery():
    file = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512))  # Model input size
    img_array = np.expand_dims(img/255.0, axis=0)
    
    # Prediction
    prediction = model.predict(img_array)[0][0]
    
    # Explainability
    heatmap = generate_gradcam(model, img_array)
    _, heatmap_img = cv2.threshold(heatmap, 0.5, 255, cv2.THRESH_BINARY)
    
    # Save visualization
    plt.imshow(img)
    plt.imshow(heatmap, alpha=0.5)
    plt.savefig('heatmap.png')
    
    return jsonify({
        'authenticity': float(1 - prediction),
        'tamper_confidence': float(prediction),
        'heatmap': 'heatmap.png'
    })

# ====================
# Training Pipeline
# ====================

def train_model(dataset_path, epochs=20):
    # Load dataset (CASIA v2/CoMoFoD structure)
    authentic_imgs = load_images(f'{dataset_path}/authentic')
    forged_imgs = load_images(f'{dataset_path}/forged')
    
    X = np.concatenate([authentic_imgs, forged_imgs])
    y = np.array([0]*len(authentic_imgs) + [1]*len(forged_imgs))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Model training
    model = create_forgery_detection_model()
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print(f"F1 Score: {f1_score(y_test, y_pred > 0.5):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
    
    # Save model
    model.save('forgery_detection_model.h5')

if __name__ == '__main__':
    # Train model
    # train_model('/path/to/dataset')
    
    # Start Flask app
    app.run(port=5000, threaded=False)
