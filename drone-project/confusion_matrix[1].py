import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "CNN method", "gesture_cnn[1].h5")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_shapes")

IMG_SIZE = 128
BATCH_SIZE = 32

# ---------------- Load model ----------------
model = load_model(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

# ---------------- Data Generator (Validation only) ----------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # IMPORTANT
)

# Class names in correct order
class_indices = val_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Classes:", class_names)

# ---------------- Predict ----------------
print("🔮 Predicting on validation set...")
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

# ---------------- Reports ----------------
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix - Shape CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
