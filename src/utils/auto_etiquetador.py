"""
AUTO-ETIQUETADOR (PRE-LABELING SCRIPT) v5 (Global)
Usa un modelo YOLOv8 pre-entrenado para generar las
etiquetas automáticamente.
Esta versión lee 'work_config.yaml' para usar rutas dinámicas
y guarda las etiquetas en la carpeta 'labels' separada.
"""
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

class AutoLabeler:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        
        # --- 1. Cargar Configuración Global ---
        self.config = self._load_config() # Cargar config
        if self.config is None:
            print("❌ Error fatal: No se pudo cargar work_config.yaml. Saliendo.")
            print("💡 Ejecuta la 'Opción 1: Configurar...' primero.")
            sys.exit(1) # Salir si no hay config
        
        self.work_type = self.config.get('work_type', 'default_job')
        print(f"Auto-etiquetador configurado para el trabajo: {self.work_type}")
        # --- Fin Carga ---
        
        # --- ¡RUTAS DINÁMICAS! ---
        # 2. De dónde LEEMOS las imágenes (definido por Opción 3)
        self.images_dir = self.project_root / "data" / "processed" / "images" / "train" / self.work_type
        # 3. Dónde ESCRIBIMOS las etiquetas (definido por Opción 1)
        self.labels_dir = self.project_root / "data" / "processed" / "labels" / "train" / self.work_type
        
        # Crear la carpeta de etiquetas (ahora dinámica) si no existe
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        # --- FIN DE LA CORRECCIÓN ---
        
        print("Cargando modelo oficial YOLOv8s...")
        self.model = YOLO('yolov8s.pt')
        
        # Cargar las clases del proyecto (dataset.yaml) y del trabajo (work_config.yaml)
        self.my_classes_map = self.load_dataset_config()
        self.work_classes_list = self.load_work_config()

        if not self.my_classes_map or not self.work_classes_list:
            print("❌ No se pudieron cargar los archivos YAML (dataset.yaml o work_config.yaml).")
            print("💡 Asegúrate de que la Opción 1 se completó correctamente.")
            sys.exit(1)
            
        # 4. Crear el mapa de traducción dinámicamente
        self.mapping = self.create_dynamic_class_mapping()
        if not self.mapping:
            print("⚠️ Advertencia: El diccionario de traducción está vacío.")

    def _load_config(self):
        """Carga el work_config.yaml completo"""
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_dataset_config(self):
        """Carga las clases desde el dataset.yaml (la fuente de verdad)"""
        config_path = self.project_root / "configs" / "dataset.yaml"
        if not config_path.exists():
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('names')

    def load_work_config(self):
        """Carga la lista de actividades desde work_config.yaml"""
        # self.config ya fue cargado en __init__
        if not self.config:
            return None
        activities_dict = self.config.get('activities', {})
        return list(activities_dict.values())

    def create_dynamic_class_mapping(self):
        """
        Crea un diccionario que traduce las clases oficiales de YOLO (COCO)
        a nuestras clases personalizadas de forma dinámica.
        """
        print("Construyendo diccionario de traducción dinámico...")
        
        official_names = self.model.names  # {0: 'person', 63: 'laptop', ...}
        my_names_map = self.my_classes_map  # {0: 'persona', 1: 'pantalla', ...}
        
        if not my_names_map:
            return {}

        # Mapas para búsqueda rápida
        official_name_to_id = {name: id for id, name in official_names.items()}
        my_name_to_id = {name: int(id) for id, name in my_names_map.items()}
        
        # 1. Mapeo simple (inglés -> español)
        simple_map = {
            'person': 'persona',
            'laptop': 'computadora',
            'keyboard': 'teclado',
            'mouse': 'mouse',
            'cell phone': 'telefono',
            'book': 'libro_documento',
            'cup': 'taza_cafe',
            'tv': 'monitor',  # 'tv' de COCO puede ser 'monitor' o 'pantalla'
            'monitor': 'monitor', # COCO también tiene 'monitor' a veces
            'hand': 'mano',
            'screen': 'pantalla'
        }

        # 2. Revisar las clases que TÚ definiste en work_config.yaml
        dynamic_translation_map = {}
        for my_class_name in self.work_classes_list:
            if my_class_name not in my_name_to_id:
                continue # Omitir si hay desincronización

            found_match = False
            for official_name, translated_name in simple_map.items():
                if translated_name == my_class_name and official_name in official_name_to_id:
                    dynamic_translation_map[official_name] = my_class_name
                    found_match = True
            
            if not found_match and my_class_name in official_name_to_id:
                dynamic_translation_map[my_class_name] = my_class_name

        # --- Fin de la Lógica Dinámica ---
        
        final_mapping = {}  # {ID Oficial (COCO) -> ID Personalizado (Tú)}
        
        print("\nDiccionario de traducción final construido:")
        for official_name, my_name in dynamic_translation_map.items():
            if official_name in official_name_to_id and my_name in my_name_to_id:
                official_id = official_name_to_id[official_name]
                my_id = my_name_to_id[my_name]
                final_mapping[official_id] = my_id
                print(f"   ✅ Mapeado: {official_name} (ID {official_id}) -> {my_name} (ID {my_id})")
        
        if not final_mapping:
            print("   ❌ ¡El diccionario de traducción está vacío!")
            
        return final_mapping

    def run_auto_label(self):
        """Ejecuta el pre-etiquetado en todas las imágenes."""
        print(f"\n🚀 Iniciando auto-etiquetado en (imágenes): {self.images_dir}")
        print(f"   Guardando etiquetas en (labels): {self.labels_dir}")
        images = list(self.images_dir.glob("*.jpg"))
        
        if not images:
            print(f"❌ No se encontraron imágenes en '{self.images_dir}'.")
            print("💡 Ejecuta la 'Opción 3: Extraer Frames' primero.")
            return

        written_count = 0
        for i, img_path in enumerate(images):
            
            # El nombre del .txt será el mismo (ej. 'interval_000000.txt')
            # pero se guarda en self.labels_dir
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists() and label_path.stat().st_size > 0:
                continue # Omitir si ya existe

            results = self.model.predict(str(img_path), verbose=False, conf=0.4)
            
            lines_to_write = []
            for box in results[0].boxes:
                detected_id = int(box.cls[0]) 
                
                if detected_id in self.mapping:
                    my_class_id = self.mapping[detected_id]
                    coords = box.xywhn[0]
                    line = f"{my_class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n"
                    lines_to_write.append(line)
            
            if lines_to_write:
                with open(label_path, 'w') as f:
                    f.writelines(lines_to_write)
                written_count += 1
                print(f"[{i+1}/{len(images)}] ✅ ETIQUETADO: {img_path.name}")
            else:
                label_path.touch() # Crear archivo vacío para saber que se procesó
                print(f"[{i+1}/{len(images)}] ⚪ Omitido: {img_path.name} (No se encontraron objetos)")
        
        print("\n✅ Auto-etiquetado completado.")
        print(f"Total de archivos de etiquetas creados/actualizados: {written_count}")
        print(f"   ➡️  Guardados en: {self.labels_dir}")
        print("💡 ¡Ahora usa la 'Opción 5: Corregir etiquetas' para revisarlas!")

def main():
    labeler = AutoLabeler()
    labeler.run_auto_label()

if __name__ == "__main__":
    main()