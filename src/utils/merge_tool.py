"""
HERRAMIENTA DE FUSIÓN DE DATASETS (v1.0)
Este script fusiona un dataset descargado (en formato YOLO)
con tu dataset de proyecto existente.

Traduce automáticamente los IDs de las clases para evitar
conflictos.
"""

import yaml
import sys
from pathlib import Path
import shutil

# 1. --- Cargar Rutas y Configuración del PROYECTO PRINCIPAL ---
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    
    # Cargar work_config para saber dónde guardar los archivos
    WORK_CONFIG_PATH = PROJECT_ROOT / "configs" / "work_config.yaml"
    with open(WORK_CONFIG_PATH, 'r', encoding='utf-8') as f:
        work_config = yaml.safe_load(f)
    
    WORK_TYPE = work_config.get('work_type')
    if not WORK_TYPE:
        raise ValueError("work_type no está definido en work_config.yaml")

    # Rutas de destino (donde se guardarán los archivos fusionados)
    DEST_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train" / WORK_TYPE
    DEST_LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train" / WORK_TYPE
    DEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar clases del proyecto principal
    DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"
    with open(DATASET_CONFIG_PATH, 'r', encoding='utf-8') as f:
        main_dataset_config = yaml.safe_load(f)
    
    # Crear un mapa de nombre a ID (ej. {'persona': '0', 'teclado': '1'})
    MAIN_CLASSES_MAP = {name: str(idx) for idx, name in main_dataset_config['names'].items()}
    print(f"✅ Configuración del proyecto '{WORK_TYPE}' cargada.")

except FileNotFoundError as e:
    print(f"❌ Error: No se pudo encontrar un archivo de configuración: {e}")
    print("💡 Asegúrate de haber ejecutado la 'Opción 1: Configurar' primero.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error al cargar la configuración del proyecto: {e}")
    sys.exit(1)


def load_new_dataset_config(new_dataset_path):
    """Carga el data.yaml del dataset descargado."""
    try:
        new_config_path = new_dataset_path / "data.yaml"
        with open(new_config_path, 'r', encoding='utf-8') as f:
            new_config = yaml.safe_load(f)
        
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN! ---
        names_data = new_config.get('names')
        
        if isinstance(names_data, list):
            # Si es una LISTA (ej. ['keyboard', 'mouse']):
            # la convertimos a un DICIONARIO (ej. {'0': 'keyboard', '1': 'mouse'})
            print("INFO: El dataset nuevo usa formato de 'lista' para las clases.")
            new_classes_map = {str(idx): name for idx, name in enumerate(names_data)}
        
        elif isinstance(names_data, dict):
            # Si ya es un DICIONARIO (ej. {'0': 'keyboard'}):
            # simplemente lo usamos.
            print("INFO: El dataset nuevo usa formato de 'dict' para las clases.")
            new_classes_map = names_data
            
        else:
            print(f"❌ Error: No se puede entender el formato de 'names' en {new_config_path}")
            return None, None
        # --- FIN DE LA CORRECCIÓN ---
            
        return new_classes_map, new_config.get('train')
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró 'data.yaml' en {new_dataset_path}")
        return None, None
    except Exception as e:
        print(f"❌ Error al leer el 'data.yaml' del nuevo dataset: {e}")
        return None, None

def get_interactive_mapping(new_classes_map):
    """Pregunta al usuario cómo mapear las clases nuevas."""
    
    print("\n--- 🕵️  Paso 1: Mapeo de Clases ---")
    print("He encontrado las siguientes clases en el dataset descargado.")
    print("Por favor, dime a qué clase de TU PROYECTO corresponden.")
    print("\n Tus clases son:")
    for name, idx in MAIN_CLASSES_MAP.items():
        print(f"   [{idx}] {name}")
    
    translation_map = {} # ej. {'0_nuevo': '1_tuyo', '1_nuevo': 'ignorar'}
    
    for new_id, new_name in new_classes_map.items():
        new_id_str = str(new_id)
        print(f"\nClase nueva: '{new_name}' (ID: {new_id_str})")
        
        while True:
            action = input(f"   ¿Qué clase de tu proyecto es esta? (Escribe el nombre, ej. 'teclado', o 'ignorar'): ").strip().lower()
            
            if action == 'ignorar':
                translation_map[new_id_str] = 'ignorar'
                print(f"   OK -> Se ignorará la clase '{new_name}'.")
                break
            elif action in MAIN_CLASSES_MAP:
                target_id = MAIN_CLASSES_MAP[action]
                translation_map[new_id_str] = target_id
                print(f"   OK -> '{new_name}' (ID {new_id_str}) se traducirá a '{action}' (ID {target_id}).")
                break
            else:
                print(f"   ❌ '{action}' no es una clase válida. Intenta de nuevo.")
    
    return translation_map

def process_and_copy(new_dataset_path, train_path_str, translation_map):
    """Traduce los .txt y copia los .jpg y .txt al proyecto principal."""
    
    print("\n--- 🚀 Paso 2: Fusión de Archivos ---")
    
    # Ruta a las imágenes y etiquetas del nuevo dataset
    # La ruta en data.yaml puede ser relativa
    new_train_path = (new_dataset_path / train_path_str).resolve()
    new_images_dir = new_train_path / "images"
    new_labels_dir = new_train_path / "labels"
    
    if not new_images_dir.exists() or not new_labels_dir.exists():
        print(f"❌ Error: No se encuentran las carpetas 'images' o 'labels' en {new_train_path}")
        return

    image_files = list(new_images_dir.glob("*.jpg")) + list(new_images_dir.glob("*.png"))
    
    print(f"Se encontraron {len(image_files)} imágenes para fusionar.")
    
    copied_count = 0
    ignored_count = 0
    
    for img_path in image_files:
        label_path = new_labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            continue # Omitir imágenes sin etiqueta

        new_label_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                old_id = parts[0]
                
                # Traducir el ID
                if old_id in translation_map:
                    new_id = translation_map[old_id]
                    
                    if new_id == 'ignorar':
                        continue # No añadir esta línea
                    
                    # Reconstruir la línea con el nuevo ID
                    new_line = f"{new_id} {' '.join(parts[1:])}"
                    new_label_lines.append(new_line)
        
        # Solo copiar si el archivo final tiene al menos una etiqueta útil
        if new_label_lines:
            # 1. Copiar la imagen
            # Damos un nombre único para evitar colisiones
            dest_img_name = f"merged_{new_dataset_path.name}_{img_path.name}"
            shutil.copy(img_path, DEST_IMAGES_DIR / dest_img_name)
            
            # 2. Escribir el archivo .txt traducido
            dest_label_name = f"merged_{new_dataset_path.name}_{img_path.stem}.txt"
            with open(DEST_LABELS_DIR / dest_label_name, 'w') as f:
                f.write("\n".join(new_label_lines))
                
            copied_count += 1
        else:
            ignored_count += 1
            
    print("\n--- ✅ Fusión Completada ---")
    print(f"   {copied_count} imágenes y etiquetas fueron fusionadas en tu proyecto.")
    print(f"   {ignored_count} imágenes fueron ignoradas (no tenían clases útiles).")
    print(f"   Tus datos están listos en: {DEST_IMAGES_DIR.parent}")

def main():
    print("--- 🛠️ Herramienta de Fusión de Datasets de IA ---")
    
    # 1. Pedir la ruta al dataset descargado
    zip_path_str = input("Pega la ruta al archivo .zip que descargaste: ").strip().replace('"', '')
    zip_path = Path(zip_path_str)
    
    if not zip_path.exists() or zip_path.suffix != '.zip':
        print("❌ Error: La ruta no es un archivo .zip válido.")
        return
        
    # 2. Descomprimir automáticamente
    extract_dir = zip_path.parent / zip_path.stem
    print(f"Descomprimiendo en: {extract_dir}...")
    try:
        shutil.unpack_archive(zip_path, extract_dir)
    except Exception as e:
        print(f"❌ Error al descomprimir el .zip: {e}")
        return
        
    print("✅ Descomprimido.")
    new_dataset_path = extract_dir
    
    # 3. Cargar config del nuevo dataset
    new_classes_map, train_path_str = load_new_dataset_config(new_dataset_path)
    if new_classes_map is None:
        return
        
    # 4. Obtener mapeo del usuario
    translation_map = get_interactive_mapping(new_classes_map)
    
    # 5. Procesar y copiar
    process_and_copy(new_dataset_path, train_path_str, translation_map)
    
    print("\n💡 ¡Ahora puedes ejecutar la 'Opción 6: Entrenar' de nuevo!")
    print("   Tu IA aprenderá de tus datos Y de los datos nuevos.")

if __name__ == "__main__":
    main()