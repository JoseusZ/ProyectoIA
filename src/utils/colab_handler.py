import os
import shutil
import zipfile
import sys

def check_and_setup_colab(base_path):
    """
    Verifica si se está ejecutando en Google Colab y gestiona la carga de datos.
    :param base_path: Ruta raíz del proyecto (donde está main.py)
    """
    
    # 1. Detectar entorno Colab
    try:
        import google.colab
        IN_COLAB = True
        print("☁️ Entorno Google Colab detectado.")
    except ImportError:
        IN_COLAB = False
        print("💻 Entorno Local detectado. Saltando configuración de nube.")
        return

    if IN_COLAB:
        print("\n--- CONFIGURACIÓN DE DATASET PARA COLAB ---")
        print("Elige cómo quieres cargar tus datos:")
        print("1. Subir archivos ZIP manualmente (Uno para imágenes, uno para etiquetas)")
        print("2. Importar desde una carpeta de Google Drive")
        print("3. Omitir (Si ya cargaste los datos previamente)")
        
        opcion = input("Selecciona una opción (1/2/3): ")

        # Rutas destino (Respetando tu estructura)
        images_dest = os.path.join(base_path, 'data', 'processed', 'images')
        labels_dest = os.path.join(base_path, 'data', 'processed', 'labels')
        
        # Crear directorios si no existen
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(labels_dest, exist_ok=True)

        if opcion == '1':
            _handle_zip_upload(images_dest, labels_dest)
        elif opcion == '2':
            _handle_drive_import(base_path)
        else:
            print("⏩ Saltando carga de datos.")

def _handle_zip_upload(images_dest, labels_dest):
    from google.colab import files
    print("\n--- PASO 1: IMÁGENES ---")
    print("Por favor, sube el ZIP que contiene las carpetas 'train' y 'val' de tus IMÁGENES.")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"📦 Descomprimiendo imágenes en {images_dest}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(images_dest)
        os.remove(filename) # Limpiar
        
    print("\n--- PASO 2: ETIQUETAS (LABELS) ---")
    print("Por favor, sube el ZIP que contiene las carpetas 'train' y 'val' de tus ETIQUETAS.")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"🏷️ Descomprimiendo etiquetas en {labels_dest}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(labels_dest)
        os.remove(filename) # Limpiar
    
    print("✅ Carga manual completada.")

def _handle_drive_import(base_path):
    from google.colab import drive
    print("\n--- IMPORTAR DESDE DRIVE ---")
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    print("Ingresa la ruta de la carpeta en Drive donde tienes 'images' y 'labels'.")
    print("Ejemplo: /content/drive/MyDrive/ProyectoIA/data/processed")
    drive_path = input("Ruta en Drive: ").strip()
    
    target_dir = os.path.join(base_path, 'data', 'processed')

    if os.path.exists(drive_path):
        print(f"🚀 Copiando archivos desde Drive a {target_dir} (Esto puede tardar)...")
        # Usamos comando de sistema para copia recursiva más eficiente
        # -n evita sobrescribir si ya existen, -r recursivo
        os.system(f"cp -r -n {drive_path}/* {target_dir}/")
        print("✅ Transferencia completada.")
    else:
        print(f"❌ La ruta {drive_path} no existe en Drive.")