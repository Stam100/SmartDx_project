# utils.py
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import io

IMG_SIZE = (128, 128)  # same as training

def load_image_pil(image_file):
    """Load an uploaded image file (Streamlit file-like) into a RGB PIL.Image."""
    image = Image.open(io.BytesIO(image_file.read()))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def preprocess_pil_image(pil_img):
    """Resize and normalize image -> numpy array ready for model (batch dimension)."""
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    # ensure shape (1, H, W, 3)
    return np.expand_dims(arr, axis=0)

def load_model(path):
    """Load Keras model (SavedModel or .h5)."""
    return tf.keras.models.load_model(path)

def predict(model, preprocessed_image):
    """Return (label, confidence, probability). Label 1 -> Parasitized."""
    probs = model.predict(preprocessed_image)  # shape (1, 1) for sigmoid
    prob = float(probs.ravel()[0])
    label = "Parasitized" if prob >= 0.5 else "Uninfected"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, float(confidence), float(prob)

# Optional: simple Grad-CAM for visualization (works for small CNNs)
def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    img_array: (1, H, W, 3) normalized
    Returns: heatmap (H,W) 0-1
    """
    # Find a convolutional layer if not provided
    if last_conv_layer_name is None:
        # pick the last layer with "conv" in name
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # assuming sigmoid single output

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    # compute guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    # resize to input size
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap (H,W 0-1) on original PIL image, return PIL image."""
    img = np.array(pil_img.resize(IMG_SIZE))
    heat = np.uint8(255 * heatmap)
    heat = cv2.applyColorMap(heat, colormap)
    overlay = cv2.addWeighted(heat, alpha, img, 1-alpha, 0)
    return Image.fromarray(overlay)
