"""
Entrenador Universal para cualquier tipo de trabajo (v3.1 - Global)
Lee todos los parámetros de entrenamiento desde 'work_config.yaml'
y usa rutas dinámicas.
CORREGIDO: Traduce 'batch_size' -> 'batch' y 'image_size' -> 'imgsz'
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
        
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN! ---
        # Traducir claves incorrectas (de setup_work.py v5) a claves correctas (de YOLO)
        if 'batch_size' in training_params:
            training_params['batch'] = training_params.pop('batch_size')
            print("INFO: Clave 'batch_size' traducida a 'batch'.")
            
        if 'image_size' in training_params:
            training_params['imgsz'] = training_params.pop('image_size')
            print("INFO: Clave 'image_size' traducida a 'imgsz'.")
        # --- FIN DE LA CORRECCIÓN ---

        # 3. Sacar el 'base_model' y dejar el resto como argumentos
        base_model_name = training_params.pop('base_model', 'yolov8n.pt')
        
        # 4. Añadir parámetros esenciales (rutas, etc.)
        static_params = {
            'data': str(self.project_root / "configs" / "dataset.yaml"),
            'project': str(self.project_root / "results"),
            'name': f"{self.work_type}_model",
            'exist_ok': True,
        }
        
        # 5. Añadir defaults (personalizables si están en el config)
        training_params.setdefault('device', 0 if torch.cuda.is_available() else 'cpu')
        training_params.setdefault('workers', 0)
        training_params.setdefault('patience', 10)
        training_params.setdefault('save', True)
        training_params.setdefault('amp', True) # Activa mixed precision
        
        # 6. Combinar todos los parámetros
        final_config = {**training_params, **static_params}

        # 7. Cargar el modelo base
        print(f"Cargando modelo base: {base_model_name}")
        model = YOLO(base_model_name)
        
        print("⚙️  Configuración de Entrenamiento Final:")
        for key, value in final_config.items():
            print(f"   {key}: {value}")
        
        # 8. Entrenar
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
        labels_dir = self.project_root / "data" / "processed" / "labels" / "train" / self.work_type
        
        # Revisa si la carpeta existe Y si hay al menos un .txt adentro
        return labels_dir.exists() and any(labels_dir.glob("*.txt"))

def main():
    trainer = UniversalTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()