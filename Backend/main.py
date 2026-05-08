import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. Configuración de rutas dinámicas
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "Model")

PATH_MODEL = os.path.join(MODEL_DIR, "clasificador_edades.pkl")
PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFE = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

CATEGORIES = {0: 'Joven', 1: 'Adulto', 2: 'Viejo'}

# 2. Inicialización de la API y modelos
app = FastAPI(
    title="Clasificador de edades",
    description="API que recibe imágenes para clasificar la edad.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    print("Cargando modelo SVM y detector de rostros DNN en memoria...")
    clf = joblib.load(PATH_MODEL)
    net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)
    print("Modelos cargados exitosamente!.")
except Exception as e:
    print(f"Error al cargar los modelos, detalle: {e}")

# 3. Lógica de Negocio (Machine Learning)

def feature_extraction(img):
    img_res = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features.reshape(1, -1)

# 4. Endpoints
@app.post("/age_classifier")
async def age_classifier(image: UploadFile = File(...)):

    # Validar que el archivo sea una imágen
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Sube una imágen válida.")

    # Leer la imagen directamente desde la memoria (sin guardarla en el disco duro)
    content = await  image.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen.")

    # Preparar la imagen para el detector DNN de OpenCV
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Buscar el rostro con mayor nivel de confianza
    foundFace = False
    bestConfidence = 0
    bestBox = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 and confidence > bestConfidence:
            bestConfidence = confidence
            bestBox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            foundFace = True

    if not foundFace:
        return {
            "status": "error",
            "message": "No se detectó ningún rostro en la fotografía",
            "prediction": None
        }

    x1, y1, x2, y2 = bestBox.astype("int")
    face = img[max(0, y1):min(h, y2), max(0, x1):min(x2, w)]

    if face.size == 0:
        raise HTTPException(status_code=500, detail="Error interno al intentar recortar el rostro")

    features = feature_extraction(face)
    prediction_idx = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]

    label = CATEGORIES[prediction_idx]
    sec = float(probabilities[prediction_idx] * 100)

    return {
        "status": "ok",
        "analyzed_file": image.filename,
        "prediction": label,
        "confidence": f"{sec: .2f}%",
        "detection_details": {
            "confidence_faceDetector": f"{bestConfidence * 100:.2f}%"
        }
    }

if __name__ == "__main__":
    print("Iniciando el servidor de desarrollo...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)