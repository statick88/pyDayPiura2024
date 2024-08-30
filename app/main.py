from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Cargar el modelo
model = tf.keras.models.load_model('cifar10_model.keras')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")  # Asegúrate de que la imagen esté en formato RGB
        image = image.resize((32, 32))  # Ajustar tamaño de imagen
        image = np.array(image) / 255.0  # Normalizar
        image = np.expand_dims(image, axis=0)  # Añadir dimensión del batch

        predictions = model.predict(image)
        class_idx = np.argmax(predictions, axis=1)[0]
        return {"class_idx": int(class_idx), "confidence": float(np.max(predictions))}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to pyDay Piura 2024"}
