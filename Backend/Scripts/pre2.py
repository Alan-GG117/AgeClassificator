import os
import cv2
import hashlib
import numpy as np
import urllib.request

# Agregamos más extensiones para capturar archivos que Windows cuenta pero el script ignoraba
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
                with open(ruta, "rb") as image_file:
                    h = hashlib.md5(image_file.read()).hexdigest()
                if h in hashes:
                    os.remove(ruta)
                    duplicados += 1
                else:
                    hashes[h] = ruta
    return total_archivos_validos, duplicados

def procesar_rostros(origen, destino, net):
    if not os.path.exists(destino): os.makedirs(destino)
    conteo = 0
    
    for root, _, files in os.walk(origen):
        for f in files:
            if not f.lower().endswith(EXTENSIONES_IMAGEN): continue
            
            # Leer imagen con soporte para caracteres especiales
            img_path = os.path.join(root, f)
            img_raw = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
            
            if img is None: continue
            
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123))
            net.setInput(blob)
            detecciones = net.forward()
            
            for i in range(detecciones.shape[2]):
                confianza = detecciones[0, 0, i, 2]
                if confianza > 0.15: # Ajusta este valor si quieres ser más/menos estricto
                    box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    
                    # Recorte seguro
                    rostro = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if rostro.size > 0:
                        nombre_final = f"rostro_{os.path.basename(root)}_{f}"
                        cv2.imwrite(os.path.join(destino, nombre_final), rostro)
                        conteo += 1
                    break
    return conteo

def main():
    ruta_base = r"c:\Users\paulu\Downloads\viejo joven"
    proto, model = descargar_modelo_dnn()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    
    for cat in ['Joven', 'Adulto', 'Viejo']:
        print(f"\n--- Analizando Categoría: {cat} ---")
        path_cat = os.path.join(ruta_base, cat)
        
        # 1. Conteo y Limpieza
        total, dups = eliminar_duplicados_y_contar(path_cat)
        print(f"Archivos de imagen detectados: {total}")
        print(f"Duplicados eliminados: {dups}")
        
        # 2. Detección
        procesadas = procesar_rostros(path_cat, os.path.join(ruta_base, f"{cat}_procesado"), net)
        print(f"Rostros extraídos con éxito: {procesadas}")
