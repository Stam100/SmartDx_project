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
    """Return (label, confidence, probability). Label 0 -> Parasitized."""
    probs = model.predict(preprocessed_image)  # shape (1, 1) for sigmoid
    prob = float(probs.ravel()[0])
    label = "Uninfected" if prob >= 0.5 else "Parasitized"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, float(confidence), float(prob)

# Simple Grad-CAM for visualization 

def _find_last_conv_layer_name(model):
    """Return the name of the last Conv2D layer in the model (or raise)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Create a Grad-CAM heatmap for a model with a single scalar output (sigmoid or softmax).
    - model: a tf.keras Model
    - img_array: numpy array shape (1, H, W, 3), preprocessed exactly as model expects
    - last_conv_layer_name: optional explicit conv layer name; if None, auto-detected
    Returns:
      heatmap: 2D numpy array with values in [0,1] resized to input image size, or None on failure
    """
    try:
        if last_conv_layer_name is None:
            last_conv_layer_name = _find_last_conv_layer_name(model)

        # Build a model that maps the input to the activations of the last conv layer
        # and the model output.
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:
            # Ensure we're watching the conv layer output
            conv_outputs, predictions = grad_model(img_array)
            # Handle multiple outputs: get scalar to compute gradients against
            # If predictions is shape (1,1) or (1,), take predictions[:,0]
            if isinstance(predictions, (list, tuple)):
                pred = predictions[0]
            else:
                pred = predictions
            # If last dimension > 1 (multi-class softmax), pick the top predicted class
            if pred.shape[-1] > 1:
                class_idx = tf.argmax(pred[0])
                loss = pred[:, class_idx]
            else:
                # binary or single-output regression: use the single output
                loss = pred[:, 0]

        # Compute gradients of the loss w.r.t conv layer outputs
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        # Global average pooling of gradients over the spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map array by the corresponding gradient importance
        conv_outputs = conv_outputs[0]  # (H, W, channels)
        pooled_grads = pooled_grads[..., tf.newaxis]  # (channels, 1)

        # Compute weighted combination: channel-wise multiply then sum across channels
        weighted = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(weighted, axis=-1)

        # Relu and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap.numpy()
        if np.max(heatmap) == 0:
            return None
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        # Resize heatmap to input image size
        input_h = img_array.shape[1]
        input_w = img_array.shape[2]
        heatmap = cv2.resize(heatmap, (input_w, input_h))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    except Exception as e:
        # Fail gracefully: return None so caller can warn instead of crashing
        # print("Grad-CAM generation failed:", str(e))
        return None


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap (H,W 0-1) on original PIL image, return PIL image."""
    img = np.array(pil_img.resize(IMG_SIZE))
    heat = np.uint8(255 * heatmap)
    heat = cv2.applyColorMap(heat, colormap)
    overlay = cv2.addWeighted(heat, alpha, img, 1-alpha, 0)
    return Image.fromarray(overlay)
