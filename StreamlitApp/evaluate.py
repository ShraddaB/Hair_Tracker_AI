import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Absolute paths to models & test datasets
# -----------------------------
HAIR_DISEASE_MODEL_PATH = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\StreamlitApp\AI_Models\hair_disease_cnn_model (1).h5"
HAIR_TYPE_MODEL_PATH    = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\StreamlitApp\AI_Models\hair_type_classifier.h5"

HAIR_DISEASE_TEST_DIR   = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\datasets\Hair Diseases - Final\test"
HAIR_TYPE_TEST_DIR      = r"C:\Users\asus\Desktop\Hair_Tracker_AI-main\Hair_Tracker_AI\datasets\hair_type_test"

# -----------------------------
# Function to evaluate a model
# -----------------------------
def evaluate_model(model_path, test_dir, model_name):
    print(f"\n🧠 Loading {model_name} from: {model_path}")
    model = load_model(model_path, compile=False)

    # Auto-detect input size
    input_shape = model.input_shape[1:3]  # (height, width)
    print(f"🔍 Model expects input size: {input_shape}")

    # Compile the model (required for evaluate)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print(f"🧮 Evaluating {model_name} on test dataset...")
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"✅ {model_name} Test Accuracy: {accuracy*100:.2f}%")
    print(f"   Test Loss: {loss:.4f}")

# -----------------------------
# Evaluate both models
# -----------------------------
evaluate_model(HAIR_DISEASE_MODEL_PATH, HAIR_DISEASE_TEST_DIR, "Hair Disease Model")
evaluate_model(HAIR_TYPE_MODEL_PATH, HAIR_TYPE_TEST_DIR, "Hair Type Model")
