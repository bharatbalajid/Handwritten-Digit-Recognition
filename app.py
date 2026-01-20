import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Handwritten Digit Recognition")
st.title("✍️ Handwritten Digit Recognition")
st.write("Try to write a numeric digit")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
)

def preprocess_canvas(img):
    img = 255 - img
    img = np.where(img > 50, 255, 0).astype(np.uint8)

    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    digit = img[y0:y1+1, x0:x1+1]

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    digit = Image.fromarray(digit).resize((new_w, new_h))

    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas28[y_off:y_off+new_h, x_off:x_off+new_w] = np.array(digit)

    cy, cx = ndimage.center_of_mass(canvas28)
    canvas28 = ndimage.shift(canvas28, (14-cy, 14-cx), mode="constant")

    return canvas28.reshape(784) / 255.0, canvas28


if st.button("Predict"):
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
        img_np = np.array(img)

        processed = preprocess_canvas(img_np)
        if processed is None:
            st.warning("Please draw a digit")
        else:
            x, preview = processed
            pred = model.predict(x.reshape(1, -1))[0]

            top = np.argsort(pred)[::-1][:3]

            st.subheader("Digit Prediction")
            for i in top:
                st.write(f"{i} : {pred[i]:.6f}")

            st.image(preview, caption="28×28 MNIST-style Input", width=120)