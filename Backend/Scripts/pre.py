import os
import cv2
import hashlib
import numpy as np

def eliminar_duplicados(carpeta_base):
    """Calcula el hash MD5 y elimina duplicados buscando en todas las subcarpetas."""
    hashes_vistos = set()
    archivos_eliminados = 0
    
    if not os.path.exists(carpeta_base):
        return 0

    # os.walk recorre la carpeta principal y todas sus subcarpetas
    for root, dirs, files in os.walk(carpeta_base):
        for archivo in files:
            if not archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            ruta_archivo = os.path.join(root, archivo)
            
            with open(ruta_archivo, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            if file_hash in hashes_vistos:
                os.remove(ruta_archivo)
                archivos_eliminados += 1
            else:
                hashes_vistos.add(file_hash)
                
    return archivos_eliminados

def recortar_y_guardar_rostros(carpeta_origen, carpeta_destino):
    """Detecta rostros escarbando en subcarpetas y guarda el recorte en la carpeta procesada."""
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostros_procesados = 0
    
    if not os.path.exists(carpeta_origen):
        return 0

    for root, dirs, files in os.walk(carpeta_origen):
        for archivo in files:
            if not archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue

            ruta_archivo = os.path.join(root, archivo)
            
            # Usamos numpy para leer la imagen y evitar errores con la "ñ" en el nombre de la subcarpeta
            img_array = np.fromfile(ruta_archivo, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                
                for (x, y, w, h) in rostros:
                    rostro_recortado = img[y:y+h, x:x+w]
                    
                    # Le agregamos el nombre de la subcarpeta al archivo para evitar que se sobreescriban
                    # si casualmente dos imágenes se llamaban "image1.jpg" en diferentes subcarpetas.
                    nombre_subcarpeta = os.path.basename(root)
                    nombre_salida = f"{nombre_subcarpeta}_{archivo}"
                    ruta_salida = os.path.join(carpeta_destino, f"rostro_{nombre_salida}")
                    
                    cv2.imwrite(ruta_salida, rostro_recortado)
                    rostros_procesados += 1
                    break # Solo extrae el primer rostro detectado por imagen
                    
    return rostros_procesados

def conteo_imagenes(carpeta_base):
    """Devuelve el número de imágenes contando las que hay en todas las subcarpetas."""
    if not os.path.exists(carpeta_base):
        return 0
    
    total = 0
    for root, dirs, files in os.walk(carpeta_base):
        for archivo in files:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                total += 1
    return total

def pipeline_preprocesamiento(ruta_base):
    # Ajustado con las mayúsculas exactas que tienes en tus carpetas
    categorias = ['Joven', 'Adulto', 'Viejo']
    
    print("--- INICIANDO PREPROCESAMIENTO LOCAL (BÚSQUEDA EN SUBCARPETAS) ---\n")
    
    for categoria in categorias:
        carpeta_original = os.path.join(ruta_base, categoria)
        carpeta_procesada = os.path.join(ruta_base, f"{categoria}_procesado")
        
        if not os.path.exists(carpeta_original):
            print(f"⚠️ La carpeta '{carpeta_original}' no existe. Saltando...")
            continue
            
        print(f"Procesando categoría: {categoria}")
        
        total_inicial = conteo_imagenes(carpeta_original)
        print(f"  -> Imágenes originales encontradas: {total_inicial}")
        
        duplicados = eliminar_duplicados(carpeta_original)
        print(f"  -> Duplicados eliminados: {duplicados}")
        
        rostros_extraidos = recortar_y_guardar_rostros(carpeta_original, carpeta_procesada)
        print(f"  -> Rostros detectados y recortados: {rostros_extraidos}")
        
        total_final = conteo_imagenes(carpeta_procesada)
        print(f"  -> Total de rostros finales listos: {total_final}\n")