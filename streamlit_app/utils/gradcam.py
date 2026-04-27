import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name):
    """
    Generate a Grad-CAM heatmap highlighting image regions that impacted the
    model's prediction the most.

    Parameters:
    img_array: np.ndarray
        Preprocessed image with shape (1, 224, 224, 3)
    model: tf.keras.Model
        Full classification model (base CNN + classification head)
    base_model: tf.keras.Model
        Base CNN model (ResNet50)
    last_conv_layer_name: str
        Name of the last convolutional layer in the base model

    Returns:
    np.ndarray
        Normalized heatmap with values between 0 and 1.
    """

    # Create a sub-model that outputs the final feature map from the frozen base
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)

    # Create a sub-model that takes the feature map and generates a final prediction
    conv_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = model.layers[-2](conv_input)  # Global Average Pooling layer
    x = model.layers[-1](x)  # Dense classification layer
    classifier_model = tf.keras.Model(conv_input, x)

    # Use GradientTape to record operations
    with tf.GradientTape() as tape:
        last_conv_layer_output = conv_model(img_array)
        tape.watch(last_conv_layer_output)

        # Pass the feature map to the classifier model to get a prediction
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, 0]

    # Calculate gradients of the top predicted class with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Average the gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important" it is
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def superimpose_heatmap(display_img, heatmap, alpha=0.4):
    """
    Superimposes a generated Grad-CAM heatmap onto the original display image.

    Parameters:
    display_img: np.ndarray
        The raw image array ready for display (scaled 0-1 or 0-255).
    heatmap: np.ndarray
        The generated 2D Grad-CAM heatmap (0 to 1).
    alpha: float
        The opacity of the heatmap overlay.

    Returns:
    PIL.Image
        The final composite image ready to be rendered in the UI.
    """

    # Create the overlay colormap
    heatmap_resized = np.uint8(255 * heatmap)
    jet = plt.colormaps['jet']
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]

    # Resize heatmap to match the original image dimensions
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((display_img.shape[1], display_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Ensure display_img is in the correct format (0-1) for mathematical addition
    if np.max(display_img) > 1.0:
        display_img = display_img / 255.0

    # Superimpose the heatmap over the original image
    superimposed_img = jet_heatmap * alpha + (display_img * 255)

    # Convert back to a PIL image object so Streamlit can render it
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)