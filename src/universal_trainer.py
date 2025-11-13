"""
Entrenador Universal para cualquier tipo de trabajo (v3 - Global)
Lee todos los parámetros de entrenamiento desde 'work_config.yaml'
y usa rutas dinámicas.
"""
import torch
from pathlib import Path
import yaml
import sys
from ultralytics import YOLO

class UniversalTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self.load_config()
        if self.config is None:
            sys.exit(1) # Salir si no hay config
            
        self.work_type = self.config.get('work_type', 'default_job')
    
    def load_config(self):
        """Carga la configuración del trabajo"""
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            print("❌ Error: 'configs/work_config.yaml' no encontrado.")
            print("💡 Ejecuta la 'Opción 1: Configurar...' primero.")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run_training(self):
        """
        Entrenamiento personalizable basado en 'work_config.yaml'
        """
        print(f"🚀 ENTRENAMIENTO PARA: {self.work_type.upper()}")
        print("=" * 50)
        
        # 1. Verificar datos
        if not self.check_training_data():
            print(f"❌ No se encontraron archivos de etiquetas (.txt) en:")
            print(f"   data/processed/labels/train/{self.work_type}")
            print("💡 Ejecuta primero las Opciones 3, 4 y 5.")
            return
        
        # 2. Cargar parámetros de entrenamiento desde el config
        training_params = self.config.get('training', {})
        
        # Sacar el 'base_model' y dejar el resto como argumentos
        base_model_name = training_params.pop('base_model', 'yolov8n.pt')
        
        # 3. Añadir parámetros esenciales (rutas, etc.)
        #    Estos NO se pueden personalizar para evitar errores.
        static_params = {
            'data': str(self.project_root / "configs" / "dataset.yaml"),
            'project': str(self.project_root / "results"),
            'name': f"{self.work_type}_model",
            'exist_ok': True,
        }
        
        # 4. Añadir defaults (personalizables si están en el config)
        #    Si ya los pusiste en work_config.yaml, se usarán esos.
        training_params.setdefault('device', 0 if torch.cuda.is_available() else 'cpu')
        training_params.setdefault('workers', 0)
        training_params.setdefault('patience', 10)
        training_params.setdefault('save', True)
        training_params.setdefault('amp', True) # Activa mixed precision
        
        # 5. Combinar todos los parámetros
        #    Los 'static_params' (rutas) sobreescriben cualquier cosa
        #    para asegurar que el entrenamiento funcione.
        final_config = {**training_params, **static_params}

        # 6. Cargar el modelo base
        print(f"Cargando modelo base: {base_model_name}")
        model = YOLO(base_model_name)
        
        print("⚙️  Configuración de Entrenamiento Final:")
        for key, value in final_config.items():
            print(f"   {key}: {value}")
        
        # 7. Entrenar
        print("\n🎯 Iniciando entrenamiento...")
        try:
            results = model.train(**final_config)
            print("✅ ENTRENAMIENTO COMPLETADO!")
            print(f"🎉 Modelo guardado en: {self.project_root / 'results' / final_config['name']}")
            return results
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            print("💡 Revisa que tu 'dataset.yaml' esté correcto (Opción 1).")
            print("💡 Si da error de 'polars', ejecuta:")
            print("   pip uninstall polars")
            print("   pip install \"polars[rtcompat]\"")
    
    def check_training_data(self):
        """
        Verifica que existan datos para entrenar (archivos .txt)
        EN LA CARPETA DINÁMICA 'labels/train/{work_type}'
        """
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN GLOBAL! ---
        labels_dir = self.project_root / "data" / "processed" / "labels" / "train" / self.work_type
        
        # Revisa si la carpeta existe Y si hay al menos un .txt adentro
        return labels_dir.exists() and any(labels_dir.glob("*.txt"))

def main():
    trainer = UniversalTrainer()
    trainer.run_training() # Se renombró de 'quick_train'

if __name__ == "__main__":
    main()