# ============================================================
# Streamlit Demo — Pneumonia Detection from Chest X-Rays
# Run locally with: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ── Load model (cached so it only loads once, not on every interaction) ──
@st.cache_resource
def get_model():
    model = load_model('pneumonia_model.keras')
    return model

model = get_model()

class_names = ['No Pneumonia', 'Yes Pneumonia']


# ── Grad-CAM helper functions ───────────────────────────────
def get_gradcam_components(model):
    """Build the flat grad-cam model once (cached separately from main model)."""
    vgg_submodel = model.get_layer('vgg16')
    flat_input = vgg_submodel.input
    flat_vgg_output = vgg_submodel.output

    extra_layers = [l for l in model.layers if l.name in ['flatten_6', 'dense_6']]
    x = flat_vgg_output
    for layer in extra_layers:
        x = layer(x)

    flat_model = tf.keras.models.Model(inputs=flat_input, outputs=x)
    grad_model = tf.keras.models.Model(
        inputs=flat_model.input,
        outputs=[flat_model.get_layer('block5_conv3').output, flat_model.output]
    )
    return grad_model


@st.cache_resource
def cached_grad_model():
    return get_gradcam_components(model)


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
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


# ── UI ───────────────────────────────────────────────────────
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

    # ── Display results side by side ──
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Original X-ray", use_container_width=True)
    with col2:
        st.image(overlaid, caption="Grad-CAM — model attention", use_container_width=True)

    # ── Prediction result ──
    if pred_class == 1:
        st.error(f"**Prediction: {class_names[pred_class]}** ({confidence:.1f}% confidence)")
    else:
        st.success(f"**Prediction: {class_names[pred_class]}** ({confidence:.1f}% confidence)")

    # ── Probability breakdown ──
    st.subheader("Confidence breakdown")
    st.progress(float(probs[0]), text=f"No Pneumonia: {probs[0]*100:.1f}%")
    st.progress(float(probs[1]), text=f"Pneumonia: {probs[1]*100:.1f}%")

    with st.expander("How to read the Grad-CAM heatmap"):
        st.markdown(
            "Red/yellow regions show where the model focused most when making "
            "its prediction. Blue regions had little influence. This does not "
            "confirm specific clinical findings (e.g. consolidation) — it only "
            "shows the model's attention pattern, used here as a sanity check "
            "against the model relying on irrelevant image artifacts."
        )
else:
    st.info("👆 Upload a chest X-ray image to get started")

st.markdown("---")
st.caption(
    "Model: VGG16 transfer learning · Test AUC-ROC: 0.959 · "
    "[GitHub repo](#) · Built as part of an AI for Medical Imaging coursework project"
)