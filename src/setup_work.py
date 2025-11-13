"""
Configuración interactiva para personalizar la IA (v6 - Avanzado)
Permite configurar los hiperparámetros de entrenamiento.
"""
import yaml
from pathlib import Path

class WorkConfigurator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "configs" / "work_config.yaml"
        
    def _get_user_input(self, prompt, default):
        """Función de ayuda para obtener entrada con valor por defecto."""
        val = input(f"   {prompt} (default: {default}): ").strip()
        return val or default

    def interactive_setup(self):
        """Configuración interactiva del tipo de trabajo y entrenamiento"""
        print("🎯 CONFIGURADOR DE ANÁLISIS DE PRODUCTIVIDAD (v6)")
        print("=" * 50)
        
        # 1. Tipo de trabajo
        work_type = input("¿Qué tipo de trabajo quieres analizar? (ej: programador, cocinero): ").strip().lower()
        if not work_type:
            work_type = "default_job"
        
        # 2. Clases de YOLO
        print("\n📝 Define las clases de YOLO (escribe 'fin' para terminar):")
        activities = ["persona"]
        print("   (Clase 0: 'persona' se añade automáticamente)")
        print("💡 Sugeridas: pantalla, teclado, mouse, mano, cuchillo, sarten, etc.")
        
        i = 1
        while True:
            activity = input(f"   Clase {i}: ").strip().lower()
            if activity == 'fin': break
            if activity and activity not in activities:
                activities.append(activity)
                i += 1
            elif not activity: break
                
        # 3. Módulos de MediaPipe
        print("\n🤖 ¿Qué módulos de MediaPipe quieres activar?")
        print("   (Escribe los números separados por comas, ej: 1,2)")
        mp_options = {'1': 'hands', '2': 'pose', '3': 'face_mesh'}
        print("   1: Hands (Manos)")
        print("   2: Pose (Postura/Cuerpo)")
        print("   3: Face Mesh (Malla Facial)")
        
        mp_choices_str = input("   Módulos a activar (default: 1,2): ").strip()
        mp_modules_to_load = []
        if mp_choices_str:
            for choice in mp_choices_str.split(','):
                if choice.strip() in mp_options:
                    mp_modules_to_load.append(mp_options[choice.strip()])
        
        if not mp_modules_to_load: # Default si el usuario no elige nada
             mp_modules_to_load = ['hands', 'pose']
        
        # --- ¡NUEVA MEJORA DE PERSONALIZACIÓN! ---
        # 4. Configuración de Entrenamiento
        print("\n⚙️  Configura los parámetros de entrenamiento (Opción 6):")
        
        # 4a. Modelo Base
        print("   Modelo Base (ej: n, s, m, l, x)")
        model_size = self._get_user_input("Tamaño de modelo (n=nano, m=medium)", "n")
        base_model = f"yolov8{model_size.lower()}.pt"
        
        # 4b. Hiperparámetros
        epochs = int(self._get_user_input("Épocas de entrenamiento", "100"))
        batch_size = int(self._get_user_input("Batch Size (memoria v-ram) (ej: 4, 8, 16)", "4"))
        image_size = int(self._get_user_input("Resolucion de imagen (ej: 416, 640)", "416"))
        
        training_config = {
            'base_model': base_model,
            'epochs': epochs,
            'batch_size': batch_size, # El script v3.1 lo traducirá a 'batch'
            'image_size': image_size, # El script v3.1 lo traducirá a 'imgsz'
            'patience': 20 # Añadido un default extra
        }
        # --- FIN DE LA MEJORA ---

        # 5. Crear configuración final
        config = {
            'project_name': f'analisis_{work_type}',
            'work_type': work_type,
            'activities': {str(idx): activity for idx, activity in enumerate(activities)},
            'mediapipe_modules': mp_modules_to_load,
            'metrics': [
                'tiempo_activo', 'cambios_actividad', 
                'eficiencia_trabajo', 'tiempo_herramientas'
            ],
            'training': training_config # <-- Se usa la config avanzada
        }
        
        # Guardar configuración
        self.save_config(config)
        self.generate_dataset_config(activities, work_type)
        
        print(f"\n✅ Configuración guardada para: {work_type}")
        print(f"📊 Clases YOLO: {activities}")
        print(f"🤖 Módulos MediaPipe: {mp_modules_to_load}")
        print(f"⚙️  Entrenamiento: {epochs} épocas, modelo {base_model}, batch {batch_size}, imgsz {image_size}")
        
        return config
    
    def save_config(self, config):
        """Guarda la configuración en YAML"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, indent=2)
    
    def generate_dataset_config(self, activities, work_type):
        """Genera el dataset.yaml (v5)"""
        data_root = str(self.project_root.resolve() / "data" / "processed")
        
        dataset_config = {
            'path': data_root,
            'train': f'images/train/{work_type}',
            'val': f'images/train/{work_type}',
            'nc': len(activities),
            'names': {str(i): activity for i, activity in enumerate(activities)}
        }
        
        dataset_path = self.project_root / "configs" / "dataset.yaml"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, allow_unicode=True, indent=2)
        
        print(f"📁 Dataset config (dataset.yaml) generado.")

def main():
    configurator = WorkConfigurator()
    configurator.interactive_setup()

if __name__ == "__main__":
    main()