"""
Configuración interactiva para personalizar la IA (v5 - Global)
"""
import yaml
from pathlib import Path

class WorkConfigurator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "configs" / "work_config.yaml"
        
    def interactive_setup(self):
        """Configuración interactiva del tipo de trabajo"""
        print("🎯 CONFIGURADOR DE ANÁLISIS DE PRODUCTIVIDAD (v5)")
        print("=" * 50)
        
        # 1. Tipo de trabajo
        work_type = input("¿Qué tipo de trabajo quieres analizar? (ej: programador, cocinero): ").strip().lower()
        if not work_type:
            work_type = "default_job" # Evitar 'programador' como default
        
        # 2. Actividades específicas (Clases de YOLO)
        print("\n📝 Define las actividades a detectar (clases de YOLO):")
        activities = ["persona"]
        print("   (Clase 0: 'persona' se añade automáticamente)")
        print("💡 Sugeridas: pantalla, teclado, mouse, mano, cuchillo, sarten, etc.")
        
        i = 1
        while True:
            activity = input(f"Clase {i}: ").strip().lower()
            if activity == 'fin':
                break
            if activity and activity not in activities:
                activities.append(activity)
                i += 1
            elif not activity:
                break
                
        # --- ¡NUEVA MEJORA GLOBAL! ---
        # 3. Módulos de MediaPipe
        print("\n🤖 ¿Qué módulos de MediaPipe quieres activar?")
        print("   (Escribe los números separados por comas, ej: 1,2)")
        mp_options = {'1': 'hands', '2': 'pose', '3': 'face_mesh'}
        print("   1: Hands (Manos)")
        print("   2: Pose (Postura/Cuerpo)")
        print("   3: Face Mesh (Malla Facial)")
        
        mp_choices_str = input("Módulos a activar (default: 1,2): ").strip()
        mp_modules_to_load = []
        if mp_choices_str:
            for choice in mp_choices_str.split(','):
                if choice.strip() in mp_options:
                    mp_modules_to_load.append(mp_options[choice.strip()])
        
        if not mp_modules_to_load: # Default si el usuario no elige nada
             mp_modules_to_load = ['hands', 'pose']
        # --- FIN DE LA MEJORA ---

        # 4. Crear configuración
        config = {
            'project_name': f'analisis_{work_type}',
            'work_type': work_type,
            'activities': {str(idx): activity for idx, activity in enumerate(activities)},
            'mediapipe_modules': mp_modules_to_load, # <-- NUEVO
            'metrics': [
                'tiempo_activo',
                'cambios_actividad', 
                'eficiencia_trabajo',
                'tiempo_herramientas'
            ],
            'training': {
                'base_model': 'yolov8n.pt',
                'epochs': 50,
                'image_size': 416,
                'batch_size': 4
            }
        }
        
        # Guardar configuración
        self.save_config(config)
        # Generar el dataset.yaml con las rutas dinámicas
        self.generate_dataset_config(activities, work_type)
        
        print(f"\n✅ Configuración guardada para: {work_type}")
        print(f"📊 Clases YOLO: {activities}")
        print(f"🤖 Módulos MediaPipe: {mp_modules_to_load}")
        
        return config
    
    def save_config(self, config):
        """Guarda la configuración en YAML"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, indent=2)
    
    def generate_dataset_config(self, activities, work_type):
        """
        Genera el archivo dataset.yaml automáticamente (v5)
        ¡Usa la estructura de carpetas images/ y labels/
        Y el work_type dinámico!
        """
        
        # Ruta base de los datos procesados
        data_root = str(self.project_root.resolve() / "data" / "processed")
        
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN GLOBAL! ---
        # Las rutas ahora son dinámicas basadas en el 'work_type'
        dataset_config = {
            'path': data_root,
            'train': f'images/train/{work_type}', # <-- RUTA DINÁMICA
            'val': f'images/train/{work_type}',   # <-- RUTA DINÁMICA
            
            'nc': len(activities),
            'names': {str(i): activity for i, activity in enumerate(activities)}
        }
        
        dataset_path = self.project_root / "configs" / "dataset.yaml"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, allow_unicode=True, indent=2)
        
        print(f"📁 Dataset config (dataset.yaml) generado CORRECTAMENTE.")
        print(f"   ➡️  Imágenes: {data_root}/images/train/{work_type}")
        print(f"   ➡️  Etiquetas: {data_root}/labels/train/{work_type}")

def main():
    configurator = WorkConfigurator()
    configurator.interactive_setup()

if __name__ == "__main__":
    main()