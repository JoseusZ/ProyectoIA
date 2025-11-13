"""
Configuración interactiva para personalizar la IA
"""
import yaml
from pathlib import Path

class WorkConfigurator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "configs" / "work_config.yaml"
        
    def interactive_setup(self):
        """Configuración interactiva del tipo de trabajo"""
        print("🎯 CONFIGURADOR DE ANÁLISIS DE PRODUCTIVIDAD")
        print("=" * 50)
        
        # 1. Tipo de trabajo
        work_type = input("¿Qué tipo de trabajo quieres analizar? (ej: programador, diseñador, cocinero): ").strip().lower()
        if not work_type:
            work_type = "programador"
        
        # 2. Actividades específicas
        print("\n📝 Define las actividades a detectar (escribe 'fin' para terminar):")
        activities = ["persona"]  # Siempre incluir persona
        
        print("💡 Actividades sugeridas: computadora, teclado, mouse, telefono, libro, taza")
        
        i = 1
        while True:
            activity = input(f"Actividad {i}: ").strip()
            if activity.lower() == 'fin':
                break
            if activity and activity not in activities:
                activities.append(activity)
                i += 1
            elif not activity:
                break
        
        # 3. Crear configuración
        config = {
            'project_name': 'analisis_productividad',
            'work_type': work_type,
            'activities': {str(idx): activity for idx, activity in enumerate(activities)},
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
        self.generate_dataset_config(activities)
        
        print(f"\n✅ Configuración guardada para: {work_type}")
        print(f"📊 Actividades a detectar: {len(activities)}")
        for idx, activity in enumerate(activities):
            print(f"   {idx}: {activity}")
        print(f"📁 Archivo: {self.config_path}")
        
        return config
    
    def save_config(self, config):
        """Guarda la configuración en YAML"""
        self.config_path.parent.mkdir(exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, indent=2)
    
    def generate_dataset_config(self, activities):
        """Genera el archivo dataset.yaml automáticamente"""
        dataset_config = {
            'path': str(self.project_root / "data"),
            'train': 'processed/images/train',
            'val': 'processed/images/val',
            'nc': len(activities),
            'names': {str(i): activity for i, activity in enumerate(activities)}
        }
        
        dataset_path = self.project_root / "configs" / "dataset.yaml"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, allow_unicode=True, indent=2)
        
        print(f"📁 Dataset config generado: {dataset_path}")

def main():
    configurator = WorkConfigurator()
    configurator.interactive_setup()

if __name__ == "__main__":
    main()