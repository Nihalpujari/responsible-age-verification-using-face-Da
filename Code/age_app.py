"""
Age Group Classifier — Streamlit app
Run:  streamlit run age_app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# ---------------- config ----------------
MODEL_PATH = Path("/home/jovyan/vault/face_project/data/"
                  "age_classifier_mobilenetv2_224_recipe2.keras")   # v7
BANDS = ["0-2", "3-9", "10-19", "20-29", "30-39",
         "40-49", "50-59", "60-69", "70+"]
IMG_SIZE = 224
LOW_RELIABILITY = {"30-39", "40-49", "50-59"}     # recall < 40% in the audit

st.set_page_config(page_title="Age Classifier", page_icon="🎂", layout="centered")

# ---------------- model (loads once) ----------------
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    def has_rescaling(model):            # v7 bakes Rescaling in -> wants [0,255]
        for l in model.layers:
            if isinstance(l, tf.keras.layers.Rescaling):
                return True
            if isinstance(l, tf.keras.Model) and has_rescaling(l):
                return True
        return False
    return m, has_rescaling(m)

model, NEEDS_RAW = load_model()

def predict(pil_img):
    x = np.asarray(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                  Image.BILINEAR), np.float32)
    if not NEEDS_RAW:                    # models without Rescaling want [0,1]
        x = x / 255.0
    probs = model.predict(x[None], verbose=0)[0]
    return probs, np.argsort(probs)[::-1]

# ---------------- UI ----------------
st.title("🎂 Age Group Classifier")
st.caption(f"MobileNetV2 · 9 age bands · UTKFace · "
           f"input {'[0,255]' if NEEDS_RAW else '[0,1]'} (auto-detected)")

mode = st.radio("Input method", ["📁 Upload", "📷 Camera"], horizontal=True)

img = None
if mode == "📁 Upload":
    up = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
else:
    shot = st.camera_input("Take a photo")
    if shot:
        img = Image.open(shot)

if img is not None:
    probs, order = predict(img)
    top, conf = int(order[0]), float(probs[order[0]])
    band = BANDS[top]

    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="input", use_container_width=True)
    with c2:
        st.metric("Predicted age band", band, f"{conf:.0%} confidence")
        st.write(f"**2nd guess:** {BANDS[int(order[1])]} "
                 f"({probs[order[1]]:.0%})")
        if band in LOW_RELIABILITY:
            st.warning(f"'{band}' is in