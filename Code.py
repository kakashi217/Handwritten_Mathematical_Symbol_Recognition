import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

# Parameters
data_dir = '/kaggle/input/mathsymbolsmedium/extracted_images'
img_size = 28
batch_size = 32

# Get class folder names and create class_to_idx mapping
class_names = sorted([folder for folder in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, folder))])
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

print("Class names:", class_names)
print("Class to index mapping:", class_to_idx)

# Collect all image paths and labels
image_paths = []
labels = []

for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        # Verify image can be read
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_paths.append(img_path)
            labels.append(class_name)

print(f"Total images found: {len(image_paths)}")

# Convert labels to indices
label_indices = [class_to_idx[label] for label in labels]

# Split into train and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, label_indices, test_size=0.2, random_state=42, stratify=label_indices
)

# Function to load and preprocess image
def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Data augmentation function for training
def augment_image(image, label):
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# Apply preprocessing and augmentation
train_dataset = (train_dataset
                 .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .shuffle(buffer_size=1000)
                 .batch(batch_size)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

print(f"Training samples: {len(train_paths)}, Testing samples: {len(test_paths)}")

# ============================================================================
# PART 2: MODEL BUILDING AND TRAINING
# ============================================================================

# Build CNN Model
def create_cnn_model(num_classes, input_shape=(28, 28, 1)):
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
num_classes = len(class_names)
model = create_cnn_model(num_classes)

# Display model architecture
model.summary()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("\n" + "="*50)
print("Starting Training...")
print("="*50 + "\n")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# PART 3: EVALUATION AND VISUALIZATION
# ============================================================================

# Evaluate model on test set
print("\n" + "="*50)
print("Evaluating Model on Test Set...")
print("="*50 + "\n")

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# Generate predictions
y_true = []
y_pred = []

print("\n" + "="*50)
print("Generating Predictions...")
print("="*50 + "\n")

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print classification report
print("\nClassification Report:")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(y_true, y_pred, class_names)

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("="*50)
for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
    print(f"{class_name}: {class_accuracy:.4f}")

# Save the model
model.save('final_symbol_detection_model.keras')
print("\n" + "="*50)
print("Model saved as 'final_symbol_detection_model.keras'")
print("="*50)

# Function to predict single image
def predict_image(image_path, model, class_names, img_size=28):
    """Predict the class of a single image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], confidence

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
