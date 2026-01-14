import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== CHANGE THESE TWO LINES ACCORDING TO WHICH MODEL YOU'RE TESTING ====
model_path = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\StreamlitApp\hair_disease_cnn_model_compatible.h5"
test_dir = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\datasets\Hair Diseases - Final\test"
# ========================================================================

# Image size should match what model was trained on (likely 128x128 or 224x224)
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load model
print(f"🧠 Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Prepare test data
print(f"📂 Loading test data from: {test_dir}")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # categorical for multi-class
    shuffle=False
)

# Evaluate model
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
