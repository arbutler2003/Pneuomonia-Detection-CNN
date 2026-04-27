import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_and_preprocess_image(uploaded_file, target_size=(224, 224), crop_percent=0.12):
    """
    Standardizes user-uploaded images to match the training pipeline.
    """
    # Convert Streamlit uploaded file to PIL then to a grey-scale Numpy Array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, None

    # 12% crop
    height, width = img.shape
    y_crop = int(height * crop_percent)
    x_crop = int(width * crop_percent)
    cropped_img = img[y_crop: height - int(y_crop * 0.5), x_crop: width - x_crop]

    # Resize and convert to RGB (ResNet50 expects 3 channels)
    resized_img = cv2.resize(cropped_img, target_size)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

    # ResNet50 Specific Preprocessing (Zero-centering & BGR conversion)
    # Expand dims to create batch size of 1 for the model
    img_array = np.expand_dims(rgb_img, axis=0).astype("float32")
    preprocessed_img = preprocess_input(img_array)

    return rgb_img, preprocessed_img