import os
import shutil
import zipfile
import sys
# Importamos tqdm para las barras de carga
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
        print("1. Cargar ZIPS (Subir manualmente a la barra lateral)")
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
            _handle_manual_zip(images_dest, labels_dest)
        elif opcion == '2':
            _handle_drive_import(base_path)
        else:
            print("⏩ Saltando carga de datos.")

def _fix_nested_structure(target_dir):
    """
    Verifica si el ZIP creó una carpeta extra contenedora y mueve los archivos
    a la raíz correcta si es necesario.
    Ejemplo: Convierte 'images/MiDataset/train' -> 'images/train'
    """
    # Listar contenido ignorando archivos ocultos del sistema (__MACOSX, .DS_Store)
    content = [c for c in os.listdir(target_dir) if not c.startswith('.') and not c.startswith('__')]
    
    # Caso 1: Si hay una sola carpeta y no es 'train' ni 'val', probablemente sea un contenedor
    if len(content) == 1:
        nested_folder_name = content[0]
        nested_folder_path = os.path.join(target_dir, nested_folder_name)
        
        if os.path.isdir(nested_folder_path) and nested_folder_name not in ['train', 'val']:
            print(f"🔧 Detectada carpeta anidada '{nested_folder_name}'. Corrigiendo estructura...")
            
            # Mover todo el contenido de la subcarpeta a la carpeta principal
            for item in os.listdir(nested_folder_path):
                src_item = os.path.join(nested_folder_path, item)
                dest_item = os.path.join(target_dir, item)
                
                # Si ya existe en destino, fusionar o sobrescribir con cuidado
                if os.path.exists(dest_item):
                    if os.path.isdir(src_item):
                        # Si es directorio, usar copytree con dirs_exist_ok (Python 3.8+)
                        shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
                        shutil.rmtree(src_item)
                    else:
                        shutil.move(src_item, dest_item)
                else:
                    shutil.move(src_item, dest_item)
            
            # Eliminar la carpeta contenedora vacía
            os.rmdir(nested_folder_path)
            print("✅ Estructura corregida: Archivos movidos a la raíz.")

def _extract_with_progress(zip_path, extract_to, description):
    """
    Función auxiliar para descomprimir mostrando barra de progreso.
    """
    try:
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
        return True
    except zipfile.BadZipFile:
        print(f"❌ Error: El archivo {zip_path} no es un ZIP válido o está corrupto.")
        return False
    except Exception as e:
        print(f"❌ Error inesperado al descomprimir: {e}")
        return False

def _handle_manual_zip(images_dest, labels_dest):
    print("\n--- ⚠️  INSTRUCCIONES DE CARGA MANUAL ---")
    print("1. Abre la barra lateral izquierda (icono de carpeta 📁).")
    print("2. Arrastra tus dos archivos ZIP (imágenes y etiquetas) ahí.")
    print("3. Espera a que termine la rueda de carga naranja en la barra lateral.")
    print("------------------------------------------------")
    
    # --- PASO 1: IMÁGENES ---
    while True:
        zip_name = input("\nEscribe el NOMBRE EXACTO de tu zip de IMÁGENES (ej: imagenes.zip): ").strip()
        if zip_name.lower() == 'salir': return

        if os.path.exists(zip_name):
            if _extract_with_progress(zip_name, images_dest, "Procesando Imágenes"):
                # APLICAMOS LA CORRECCIÓN DE ESTRUCTURA AQUÍ
                _fix_nested_structure(images_dest)
                # Opcional: Borrar zip para ahorrar espacio
                # os.remove(zip_name) 
                break
        else:
            print(f"❌ No encuentro el archivo '{zip_name}'. Asegúrate de haberlo subido y revisa el nombre.")
            print("(Escribe 'salir' para cancelar)")

    # --- PASO 2: ETIQUETAS ---
    while True:
        zip_name = input("\nEscribe el NOMBRE EXACTO de tu zip de ETIQUETAS (ej: labels.zip): ").strip()
        if zip_name.lower() == 'salir': return

        if os.path.exists(zip_name):
            if _extract_with_progress(zip_name, labels_dest, "Procesando Etiquetas"):
                # APLICAMOS LA CORRECCIÓN DE ESTRUCTURA AQUÍ TAMBIÉN
                _fix_nested_structure(labels_dest)
                # os.remove(zip_name)
                break
        else:
            print(f"❌ No encuentro el archivo '{zip_name}'.")

    print("\n✅ Carga manual completada.")

def _copy_dir_with_progress(src_path, dest_path):
    """
    Copia un directorio recursivamente mostrando barra de progreso.
    """
    print("📊 Calculando archivos a copiar...")
    total_files = 0
    for root, dirs, files in os.walk(src_path):
        total_files += len(files)
    
    if total_files == 0:
        print("⚠️  La carpeta origen parece vacía.")
        return

    print(f"🚀 Copiando {total_files} archivos...")
    
    with tqdm(total=total_files, unit="file", desc="Transfiriendo") as pbar:
        for root, dirs, files in os.walk(src_path):
            rel_path = os.path.relpath(root, src_path)
            current_dest_dir = os.path.join(dest_path, rel_path)
            
            os.makedirs(current_dest_dir, exist_ok=True)
            
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(current_dest_dir, file)
                
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
    
    target_dir = os.path.join(base_path, 'data', 'processed')

    if os.path.exists(drive_path):
        _copy_dir_with_progress(drive_path, target_dir)
        print("\n✅ Transferencia completada.")
    else:
        print(f"❌ La ruta {drive_path} no existe en Drive.")