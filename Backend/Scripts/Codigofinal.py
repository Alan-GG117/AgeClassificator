#import cv2
#import numpy as np
#import joblib
#import os
#from skimage.feature import hog
#import tkinter as tk
#from tkinter import filedialog

## 1. Configuración de rutas y etiquetas
#MODELO_PATH = 'clasificador_edades.pkl'
#PROTO = "deploy.prototxt"
#CAFFE = "res10_300x300_ssd_iter_140000.caffemodel"
#CATEGORIAS = {0: 'Joven', 1: 'Adulto', 2: 'Viejo'}
#
#def extraer_caracteristicas(imagen_rostro):
#    """Convierte el rostro en datos numéricos (HOG)"""
#    img_res = cv2.resize(imagen_rostro, (64, 128))
#    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
#    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
#                   cells_per_block=(2, 2), visualize=False)
#    return features
#
#def seleccionar_archivo():
#    """Abre una ventana para elegir la foto con el ratón"""
#    root = tk.Tk()
#    root.withdraw() # Oculta la ventana principal de tk
#    ruta = filedialog.askopenfilename(
#        title="Selecciona una foto para clasificar",
#        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp")]
#    )
#    return ruta
#
#def clasificar():
#    # Verificar que el modelo existe
#    if not os.path.exists(MODELO_PATH):
#        print(f"Error: No se encuentra el archivo {MODELO_PATH}. ¿Ya entrenaste el modelo?")
#        return
#
#    # 1. Seleccionar la foto
#    ruta_imagen = seleccionar_archivo()
#    if not ruta_imagen:
#        print("No seleccionaste ninguna imagen.")
#        return
#
#    # 2. Cargar modelo y detector
#    clf = joblib.load(MODELO_PATH)
#    net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)
#
#    # 3. Leer imagen (con soporte para rutas de Windows)
#    img_raw = np.fromfile(ruta_imagen, np.uint8)
#    img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
#
#    if img is None:
#        print("No se pudo leer la imagen.")
#        return
#
#    h, w = img.shape[:2]
#    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123))
#    net.setInput(blob)
#    detecciones = net.forward()
#
#    rostro_encontrado = False
#    for i in range(detecciones.shape[2]):
#        confianza = detecciones[0, 0, i, 2]
#        if confianza > 0.5: # Si el detector está seguro al 50%
#            rostro_encontrado = True
#            box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
#            x1, y1, x2, y2 = box.astype("int")
#
#            # Recorte del rostro
#            rostro = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
#
#            if rostro.size > 0:
#                # Predicción con el modelo entrenado
#                features = extraer_caracteristicas(rostro).reshape(1, -1)
#                prediccion = clf.predict(features)[0]
#                probabilidades = clf.predict_proba(features)[0]
#
#                etiqueta = CATEGORIAS[prediccion]
#                seguridad = probabilidades[prediccion] * 100
#
#                print(f"\n--- ANÁLISIS COMPLETADO ---")
#                print(f"Resultado: {etiqueta}")
#                print(f"Confianza del modelo: {seguridad:.2f}%")
#
#                # Dibujar en la imagen
#                color = (0, 255, 0) if prediccion == 0 else (255, 255, 0)
#                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#                cv2.putText(img, f"{etiqueta} ({seguridad:.1f}%)", (x1, y1-10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#            break # Solo procesamos el primer rostro detectado
#
#    if not rostro_encontrado:
#        print("No se detectó ningún rostro claro en la foto.")
#    else:
#        # Mostrar la foto con el resultado
#        cv2.imshow("Resultado de la Clasificacion", img)
#        print("\nPresiona cualquier tecla en la ventana de la imagen para cerrar.")
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()