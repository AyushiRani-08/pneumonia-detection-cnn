# ============================================================
# Streamlit Demo — Pneumonia Detection from Chest X-Rays
# ============================================================

import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ── Load model from Google Drive ───────────────────────────
MODEL_PATH = 'pneumonia_model.keras'
FILE_ID = "1TJwx7F4ZS-ZEs8EPjkZCpplJjrsKIGUo"

@st.cache_resource
def get_model():
    """
    Downloads model from Google Drive if not present, then loads it.
    Falls back to manual architecture reconstruction if direct load fails.
    """
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (56 MB)..."):
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_PATH,
                quiet=False
            )

    try:
        # Strategy A: straightforward load
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        # Strategy B: reconstruct architecture and load weights
        inputs = Input(shape=(150, 150, 3), name="input_layer_20")
        vgg_base = VGG16(weights=None, include_top=False, input_shape=(150, 150, 3))
        x = vgg_base(inputs, training=False)
        x = Flatten(name="flatten_6")(x)
        outputs = Dense(2, activation="softmax", name="dense_6")(x)
        native_model = Model(inputs=inputs, outputs=outputs)
        native_model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        return native_model

model = get_model()
class_names = ['No Pneumonia', 'Yes Pneumonia']


# ── Grad-CAM ───────────────────────────────────────────────
@st.cache_resource
def cached_grad_model():
    """
    Isolates the internal feature extraction layer (block5_conv3)
    from the VGG16 backbone for Grad-CAM visualization.
    """
    vgg_backbone = model.get_layer('vgg16')
    target_layer = vgg_backbone.get_layer('block5_conv3')
    vgg_grad_model = tf.keras.models.Model(
        inputs=vgg_backbone.inputs,
        outputs=[target_layer.output, vgg_backbone.output]
    )
    return vgg_grad_model


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    flatten_layer = model.get_layer('flatten_6')
    dense_layer = model.get_layer('dense_6')

    with tf.GradientTape() as tape:
        conv_outputs, vgg_output = grad_model(img_array)
        x = flatten_layer(vgg_output)
        predictions = dense_layer(x)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_gradcam(img, heatmap, alpha=0.4):
    img_uint8 = np.uint8(255 * img) if img.max() <= 1.0 else np.uint8(img)
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)


def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img).astype('float32') / 255.0
    return img_array


# ── UI ──────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.markdown(
    "Upload a chest X-ray image to get a prediction, confidence score, "
    "and a Grad-CAM visualization showing which lung regions the model focused on."
)

st.warning(
    "⚠️ **For educational/demo purposes only.** This is a coursework project, "
    "not a medical diagnostic tool. Do not use for actual clinical decisions."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing X-ray..."):
        grad_model = cached_grad_model()
        heatmap, pred_class, probs = make_gradcam_heatmap(img_batch, grad_model)
        overlaid = overlay_gradcam(img_array, heatmap)

    confidence = probs[pred_class] * 100

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Original X-ray", use_container_width=True)
    with col2:
        st.image(overlaid, caption="Grad-CAM — model attention", use_container_width=True)

    if pred_class == 1:
        st.error(f"**Prediction: {class_names[pred_class]}** ({confidence:.1f}% confidence)")
    else:
        st.success(f"**Prediction: {class_names[pred_class]}** ({confidence:.1f}% confidence)")

    st.subheader("Confidence breakdown")
    st.progress(float(probs[0]), text=f"No Pneumonia: {probs[0]*100:.1f}%")
    st.progress(float(probs[1]), text=f"Pneumonia: {probs[1]*100:.1f}%")

    with st.expander("How to read the Grad-CAM heatmap"):
        st.markdown(
            "Red/yellow regions show where the model focused most when making "
            "its prediction. Blue regions had little influence. This does not "
            "confirm specific clinical findings — it only shows the model's "
            "attention pattern as a sanity check against spurious shortcuts."
        )
else:
    st.info("👆 Upload a chest X-ray image to get started")

st.markdown("---")
st.caption(
    "Model: VGG16 transfer learning · Test AUC-ROC: 0.959 · "
    "Built as part of an AI for Medical Imaging coursework project"
)