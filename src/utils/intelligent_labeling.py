"""
SISTEMA INTELIGENTE DE CORRECCIÓN DE ETIQUETAS (v3 - Global)
Analiza las pre-etiquetas (de rutas dinámicas)
y guía al usuario para corregirlas usando Roboflow.
"""
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime

class IntelligentLabeling:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        
        # --- 1. Cargar Configuración Global ---
        self.config = self._load_config() # Cargar config
        if self.config is None:
            print("❌ Error fatal: No se pudo cargar work_config.yaml. Saliendo.")
            print("💡 Ejecuta la 'Opción 1: Configurar...' primero.")
            sys.exit(1) # Salir si no hay config
        
        self.work_type = self.config.get('work_type', 'default_job')
        print(f"Sistema de corrección configurado para: {self.work_type}")
        # --- Fin Carga ---
        
        # --- ¡RUTAS DINÁMICAS! ---
        # De dónde LEEMOS las imágenes
        self.images_dir = self.project_root / "data" / "processed" / "images" / "train" / self.work_type
        # De dónde LEEMOS las etiquetas
        self.labels_dir = self.project_root / "data" / "processed" / "labels" / "train" / self.work_type
        # --- FIN DE LA CORRECCIÓN ---
        
        # Carga las clases dinámicamente desde dataset.yaml
        self.my_classes_map = self._load_classes_from_dataset()

    def _load_config(self):
        """Carga el work_config.yaml completo"""
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_classes_from_dataset(self):
        """Carga las clases desde dataset.yaml"""
        config_path = self.project_root / "configs" / "dataset.yaml"
        if not config_path.exists():
            print("❌ Error: 'configs/dataset.yaml' no encontrado.")
            print("💡 Ejecuta la 'Opción 1: Configurar...' primero.")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'names' not in config:
            print("❌ Error: 'dataset.yaml' no contiene una lista de 'names'.")
            return None
            
        print("✅ Clases del proyecto cargadas:")
        print(config['names'])
        return config['names']

    def check_status(self):
        """Analiza el estado actual del etiquetado."""
        print("\n" + "="*70)
        print("🏷️  SISTEMA DE CORRECCIÓN DE ETIQUETAS")
        print("="*70)
        
        if not self.my_classes_map:
            return False # Detener si no se cargaron las clases

        # --- ¡RUTAS CORREGIDAS! ---
        images = list(self.images_dir.glob("*.jpg"))
        labels = list(self.labels_dir.glob("*.txt")) # <-- Lee desde la carpeta 'labels'
        
        if not images:
            print(f"❌ No se encontraron imágenes (.jpg) en: {self.images_dir}")
            print("💡 Ejecuta la 'Opción 3: Extraer Frames' primero.")
            return False

        total_images = len(images)
        total_labels_files = len(labels)
        
        # Contar cuántos .txt no están vacíos
        pre_labeled_count = 0
        for label_file in labels: # <-- Lee desde la carpeta 'labels'
            if label_file.stat().st_size > 0:
                pre_labeled_count += 1
        
        print(f"📊 ANÁLISIS DEL DATASET ({self.work_type}):")
        print(f"   Total de Imágenes (.jpg): {total_images} (en {self.images_dir.name})")
        print(f"   Total de Archivos de Etiqueta (.txt): {total_labels_files} (en {self.labels_dir.name})")
        print(f"   Imágenes con Pre-etiquetas (de Opción 4): {pre_labeled_count}")
        print(f"   Imágenes que necesitan corrección/revisión: {total_images}")
        print("─"*70)
        return True

    def show_roboflow_instructions(self):
        """Muestra las instrucciones para Roboflow (v3 - Global)"""
        print("\n🚀 HERRAMIENTA RECOMENDADA: ROBOFLOW")
        print("Usaremos una herramienta profesional y estable.")
        
        print("\n📋 INSTRUCCIONES PARA CORREGIR ETIQUETAS:")
        print("   1.  Ve a https://roboflow.com/ y crea una cuenta (gratis).")
        print("   2.  Crea un 'Nuevo Proyecto' de 'Object Detection (Bounding Box)'.")
        print("   3.  Sube tus archivos. Arrastra y suelta **TODOS** los archivos de:")
        
        # --- ¡RUTAS CORREGIDAS! ---
        print(f"       IMÁGENES: {self.images_dir}")
        print(f"       ETIQUETAS: {self.labels_dir}")
        print("       (¡Arrastra los .jpg y los .txt juntos!)")
        
        print("\n   --- En Roboflow ---")
        print("   4.  Haz clic en 'Finish Uploading' y espera que procese.")
        print("   5.  Ve a la pestaña 'Annotate' (barra lateral).")
        print("   6.  ¡Verás tus imágenes con las cajas de la IA ya dibujadas!")
        print("   7.  En la barra derecha, renombra las clases (ej. 'class-0' -> 'persona')")
        print("       para que coincidan con tu lista:")
        
        for idx, name in self.my_classes_map.items():
            print(f"       - '{int(idx)}' -> {name}")
            
        print("\n   --- Tu Tarea ---")
        print("   8.  **CORRIGE** las cajas malas, **AJUSTA** las imprecisas.")
        print("   9.  **DIBUJA** las cajas que faltaron (¡especialmente las clases manuales!).")
        
        print("\n   --- Al Terminar ---")
        print("   10. Haz clic en 'Generate New Version' (botón verde).")
        print("   11. Sigue los pasos (puedes dejar todo como está).")
        print("   12. Al final, haz clic en 'Export' (junto a la versión de tu dataset).")
        print("   13. Elige formato 'YOLO v8' y descarga el .zip.")
        print("   14. Descomprime ese .zip. Dentro, encontrarás las carpetas 'train' y 'valid'.")
        print("   15. Copia los archivos de esas carpetas de vuelta a tu proyecto:")
        
        # --- ¡RUTAS CORREGIDAS! ---
        print(f"       IMÁGENES (.jpg) de vuelta a: {self.images_dir}")
        print(f"       ETIQUETAS (.txt) de vuelta a: {self.labels_dir}")
        
        print("\n" + "─"*70)
        print("💡 ¡Estarás listo para la 'Opción 6: Entrenar modelo'!")
        input("\nPresiona Enter para volver al menú principal...")

def main():
    """
    Función principal para el script de corrección de etiquetas (Opción 5).
    """
    labeling_system = IntelligentLabeling()
    
    # 1. Analizar estado
    if not labeling_system.check_status():
        input("\nPresiona Enter para volver al menú principal...")
        return
    
    # 2. Mostrar instrucciones de la herramienta
    labeling_system.show_roboflow_instructions()

if __name__ == "__main__":
    main()