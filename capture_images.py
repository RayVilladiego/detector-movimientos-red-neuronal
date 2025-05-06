import cv2
import os

LABEL = "saludo"  # Cambia a la clase que quieras capturar
SAVE_PATH = f"data/{LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Captura", frame)
    k = cv2.waitKey(1)
    if k % 256 == 32:  # Espacio
        img_name = f"{SAVE_PATH}/{LABEL}_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Guardada: {img_name}")
        count += 1
    elif k % 256 == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
