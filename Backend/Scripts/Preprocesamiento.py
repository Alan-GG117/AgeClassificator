import os
import cv2
import hashlib
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import joblib  # Para guardar el modelo entrenado

# Extensiones de imagen
EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.jfif', '.tiff')

def descargar_modelo_dnn():
    prototxt_path = "deploy.prototxt"
    caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
    if not os.path.exists(prototxt_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(url, prototxt_path)
    if not os.path.exists(caffemodel_path):
        url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(url, caffemodel_path)
    return prototxt_path, caffemodel_path

def eliminar_duplicados_y_contar(carpeta):
    hashes = {}
    duplicados = 0
    total_archivos_validos = 0
    for root, _, files in os.walk(carpeta):
        for f in files:
            if f.lower().endswith(EXTENSIONES_IMAGEN):
                total_archivos_validos += 1
                ruta = os.path.join(root, f)
                try:
                    with open(ruta, "rb") as image_file:
                        h = hashlib.md5(image_file.read()).hexdigest()
                    if h in hashes:
                        os.remove(ruta)
                        duplicados += 1
                    else:
                        hashes[h] = ruta
                except: continue
    return total_archivos_validos, duplicados

def extraer_caracteristicas(imagen_rostro):
    """Convierte un recorte de rostro en un vector numérico usando HOG."""
    # Redimensionar a un tamaño estándar para el modelo
    img_res = cv2.resize(imagen_rostro, (64, 128))
    # Convertir a gris
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    # Extraer HOG (características de bordes y formas)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features

def procesar_y_preparar_datos(origen, net):
    X = []
    y = []
    categorias = {'Joven': 0, 'Adulto': 1, 'Viejo': 2}
    
    for cat, label in categorias.items():
        path_cat = os.path.join(origen, cat)
        
        # --- MENSAJE DE CONTROL 1: Ver si encuentra la carpeta ---
        if not os.path.exists(path_cat):
            print(f"ALERTA: No se encontró la carpeta: {path_cat}")
            continue
            
        print(f"Procesando fotos en: {cat}...")
        archivos_encontrados = 0
        rostros_detectados = 0
        
        for root, _, files in os.walk(path_cat):
            for f in files:
                if not f.lower().endswith(EXTENSIONES_IMAGEN): continue
                archivos_encontrados += 1
                
                img_path = os.path.join(root, f)
                img_raw = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
                
                if img is None: 
                    print(f"No se pudo leer la imagen: {f}")
                    continue
                
                h, w = img.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123))
                net.setInput(blob)
                detecciones = net.forward()
                
                for i in range(detecciones.shape[2]):
                    # --- CAMBIO: Bajamos la confianza a 0.2 para detectar más rostros ---
                    confianza = detecciones[0, 0, i, 2]
                    if confianza > 0.2: 
                        box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype("int")
                        rostro = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        
                        if rostro.size > 0:
                            features = extraer_caracteristicas(rostro)
                            X.append(features)
                            y.append(label)
                            rostros_detectados += 1
                        break
        
        print(f"   -> Imágenes encontradas: {archivos_encontrados}")
        print(f"   -> Rostros detectados con éxito: {rostros_detectados}")

    return np.array(X), np.array(y)

def entrenar_modelo(X, y):
    print(f"\n--- Iniciando Entrenamiento ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #  clasificador SVM (Máquina de Soporte Vectorial)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    # Validación
    y_pred = clf.predict(X_test)
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Joven', 'Adulto', 'Viejo']))
    
    return clf

def main():
    ruta_base = r"C:\Users\chave\Desktop\viejojovenadulto\viejojovenadulto"
    proto, model = descargar_modelo_dnn()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    
    # 1. Limpieza de datos originales
    for cat in ['Joven', 'Adulto', 'Viejo']:
        path_cat = os.path.join(ruta_base, cat)
        if os.path.exists(path_cat):
            total, dups = eliminar_duplicados_y_contar(path_cat)
            print(f"Categoría {cat}: {total} archivos, {dups} duplicados eliminados.")

    # 2. Extracción de rostros y características
    X, y = procesar_y_preparar_datos(ruta_base, net)
    
    if len(X) == 0:
        print("No se encontraron rostros suficientes para entrenar.")
        return

    # 3. Entrenamiento 
    modelo_final = entrenar_modelo(X, y)
    
    # 4. Guardar modelo 
    joblib.dump(modelo_final, 'clasificador_edades.pkl')
    print("\n¡Modelo guardado como 'clasificador_edades.pkl'!")