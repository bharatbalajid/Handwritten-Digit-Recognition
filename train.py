import tensorflow as tf
import numpy as np
from scipy import ndimage
from PIL import Image

def preprocess(img):
    img = 255 - img
    img = np.where(img > 50, 255, 0).astype(np.uint8)

    coords = np.column_stack(np.where(img > 0))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    digit = img[y0:y1+1, x0:x1+1]

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    digit = Image.fromarray(digit).resize((new_w, new_h))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = np.array(digit)

    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14-cy, 14-cx), mode="constant")

    return canvas.reshape(784) / 255.0


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.array([preprocess(img) for img in x_train])
x_test  = np.array([preprocess(img) for img in x_test])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(30, activation="sigmoid"),
    tf.keras.layers.Dense(10, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=3.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=10,
    validation_data=(x_test, y_test)
)

model.save("digit_model.keras")
print("model saved")