import os
import shutil
import zipfile
import sys
# Importamos tqdm para las barras de carga. 
# En Colab suele venir instalado, si no, se instalará con pip install tqdm
from tqdm.auto import tqdm 

def check_and_setup_colab(base_path):
    """
    Verifica si se está ejecutando en Google Colab y gestiona la carga de datos.
    :param base_path: Ruta raíz del proyecto (donde está main.py)
    """
    
    # 1. Detectar entorno Colab
    try:
        import google.colab
        IN_COLAB = True
        print("☁️  Entorno Google Colab detectado.")
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

def _extract_with_progress(zip_path, extract_to, description):
    """
    Función auxiliar para descomprimir mostrando barra de progreso.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Obtenemos la lista de archivos para saber el total
        files = zip_ref.infolist()
        total_files = len(files)
        
        # Configuramos la barra de carga
        print(f"\n📂 {description}...")
        with tqdm(total=total_files, unit="file", desc="Descomprimiendo") as pbar:
            for file in files:
                zip_ref.extract(file, extract_to)
                pbar.update(1)

def _handle_zip_upload(images_dest, labels_dest):
    from google.colab import files
    
    print("\n--- PASO 1: IMÁGENES ---")
    print("Por favor, sube el ZIP que contiene las carpetas 'train' y 'val' de tus IMÁGENES.")
    # La subida (upload) tiene su propia barra nativa de Colab, no necesitamos tqdm aquí
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        # Usamos la nueva función con barra de progreso para descomprimir
        _extract_with_progress(filename, images_dest, "Procesando Imágenes")
        os.remove(filename) # Limpiar el zip
        
    print("\n--- PASO 2: ETIQUETAS (LABELS) ---")
    print("Por favor, sube el ZIP que contiene las carpetas 'train' y 'val' de tus ETIQUETAS.")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        # Usamos la nueva función con barra de progreso para descomprimir
        _extract_with_progress(filename, labels_dest, "Procesando Etiquetas")
        os.remove(filename) # Limpiar el zip
    
    print("\n✅ Carga manual completada.")

def _copy_dir_with_progress(src_path, dest_path):
    """
    Copia un directorio recursivamente mostrando barra de progreso.
    """
    # 1. Contar archivos primero para configurar la barra
    print("📊 Calculando archivos a copiar...")
    total_files = 0
    for root, dirs, files in os.walk(src_path):
        total_files += len(files)
    
    if total_files == 0:
        print("⚠️  La carpeta origen parece vacía.")
        return

    # 2. Copiar con barra
    print(f"🚀 Copiando {total_files} archivos...")
    copied_count = 0
    
    with tqdm(total=total_files, unit="file", desc="Transfiriendo") as pbar:
        for root, dirs, files in os.walk(src_path):
            # Calcular ruta relativa para replicar estructura en destino
            rel_path = os.path.relpath(root, src_path)
            current_dest_dir = os.path.join(dest_path, rel_path)
            
            # Crear directorio destino si no existe
            os.makedirs(current_dest_dir, exist_ok=True)
            
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(current_dest_dir, file)
                
                # Copiar solo si no existe (equivalente a cp -n) para ganar velocidad en reintentos
                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)
                
                pbar.update(1)

def _handle_drive_import(base_path):
    from google.colab import drive
    print("\n--- IMPORTAR DESDE DRIVE ---")
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    print("Ingresa la ruta de la carpeta en Drive donde tienes 'images' y 'labels'.")
    print("Ejemplo: /content/drive/MyDrive/ProyectoIA/data/processed")
    drive_path = input("Ruta en Drive: ").strip()
    
    # Ruta destino local en Colab
    target_dir = os.path.join(base_path, 'data', 'processed')

    if os.path.exists(drive_path):
        # Llamamos a nuestra función personalizada de copiado
        _copy_dir_with_progress(drive_path, target_dir)
        print("\n✅ Transferencia completada.")
    else:
        print(f"❌ La ruta {drive_path} no existe en Drive.")