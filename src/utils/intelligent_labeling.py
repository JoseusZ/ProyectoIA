"""
SISTEMA INTELIGENTE DE CORRECCIÓN DE ETIQUETAS (v2)
Analiza las pre-etiquetas generadas por la Opción 4
y guía al usuario para corregirlas usando Roboflow.
"""
import sys
from pathlib import Path
import yaml  # <-- Importante
import json
import random
from datetime import datetime

class IntelligentLabeling:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.images_dir = self.project_root / "data" / "processed" / "images" / "train" / "smart"
        # Carga las clases dinámicamente desde el archivo de configuración
        self.my_classes_map = self._load_classes()

    def _load_classes(self):
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

        images = list(self.images_dir.glob("*.jpg"))
        labels = list(self.images_dir.glob("*.txt"))
        
        if not images:
            print(f"❌ No se encontraron imágenes (.jpg) en: {self.images_dir}")
            print("💡 Ejecuta la 'Opción 3: Extraer Frames' primero.")
            return False

        total_images = len(images)
        total_labels_files = len(labels)
        
        # Contar cuántos .txt no están vacíos
        pre_labeled_count = 0
        for label_file in labels:
            if label_file.stat().st_size > 0:
                pre_labeled_count += 1
        
        print(f"📊 ANÁLISIS DEL DATASET:")
        print(f"   Total de Imágenes (.jpg): {total_images}")
        print(f"   Total de Archivos de Etiqueta (.txt): {total_labels_files}")
        print(f"   Imágenes con Pre-etiquetas (de Opción 4): {pre_labeled_count}")
        print(f"   Imágenes que necesitan corrección/revisión: {total_images}")
        print("─"*70)
        return True

    def show_roboflow_instructions(self):
        """Muestra las instrucciones para Roboflow."""
        print("\n🚀 HERRAMIENTA RECOMENDADA: ROBOFLOW")
        print("Las herramientas anteriores (LabelImg, Makesense) fallaron.")
        print("Usaremos una herramienta profesional y estable.")
        
        print("\n📋 INSTRUCCIONES PARA CORREGIR ETIQUETAS:")
        print("   1.  Ve a https://roboflow.com/ y crea una cuenta (gratis).")
        print("   2.  Crea un 'Nuevo Proyecto' de 'Object Detection (Bounding Box)'.")
        print("   3.  Sube tus archivos. Arrastra y suelta **TODOS** los archivos de:")
        print(f"       {self.images_dir}")
        print("       (¡Arrastra los .jpg y los .txt generados por la Opción 4 juntos!)")
        print("\n   --- En Roboflow ---")
        print("   4.  Haz clic en 'Finish Uploading' y espera que procese.")
        print("   5.  Ve a la pestaña 'Annotate' (barra lateral).")
        print("   6.  ¡Verás tus imágenes con las cajas de la IA ya dibujadas!")
        print("   7.  En la barra derecha, renombra las clases (ej. 'class-0' -> 'persona')")
        print("       para que coincidan con tu lista:")
        
        # Imprimir la lista de clases para que el usuario la vea
        for idx, name in self.my_classes_map.items():
            # Los IDs en el YAML son strings ('0'), los convertimos
            print(f"       - '{int(idx)}' -> {name}")
            
        print("\n   --- Tu Tarea ---")
        print("   8.  **CORRIGE** las cajas malas, **AJUSTA** las imprecisas.")
        print("   9.  **DIBUJA** las cajas que faltaron (¡especialmente 'mano' y 'puntero'!).")
        print("\n   --- Al Terminar ---")
        print("   10. Haz clic en 'Generate New Version' (botón verde).")
        print("   11. Sigue los pasos (puedes dejar todo como está).")
        print("   12. Al final, haz clic en 'Export' (junto a la versión de tu dataset).")
        print("   13. Elige formato 'YOLO v8' y descarga el .zip.")
        print("   14. Descomprime ese .zip. Dentro, encontrarás las carpetas 'train' y 'valid'.")
        print("   15. Copia las imágenes Y etiquetas de esa carpeta 'train'")
        print("       de vuelta a tu carpeta 'data/processed/images/train/smart'.")
        
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