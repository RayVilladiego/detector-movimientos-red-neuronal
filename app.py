import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

st.title("🧠 Detector de Movimientos por Red Neuronal")
model = tf.keras.models.load_model("../model/detector_mov.h5")

class_names = ['neutro', 'saludo', 'stop']  # Ajusta según tus clases

cam = cv2.VideoCapture(0)

stframe = st.empty()
st.write("Presiona 'q' para detener.")

while True:
    ret, frame = cam.read()
    if not ret:
        st.error("No se puede acceder a la cámara.")
        break

    resized = cv2.resize(frame, (224, 224)) / 255.0
    input_tensor = np.expand_dims(resized, axis=0)
    prediction = model.predict(input_tensor)
    label = class_names[np.argmax(prediction)]

    cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    stframe.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
