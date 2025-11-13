"""
Entrenador Universal para cualquier tipo de trabajo
"""
import torch
from pathlib import Path
import yaml
from ultralytics import YOLO

class UniversalTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self.load_config()
    
    def load_config(self):
        """Carga la configuración del trabajo"""
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            print("❌ Primero configura el proyecto con: Opción 1")
            return {'work_type': 'default', 'training': {'epochs': 50, 'image_size': 416, 'batch_size': 4}}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def quick_train(self):
        """Entrenamiento rápido y sencillo"""
        work_type = self.config.get('work_type', 'default')
        print(f"🚀 ENTRENAMIENTO PARA: {work_type.upper()}")
        print("=" * 50)
        
        # Verificar datos
        if not self.check_training_data():
            # El mensaje de error ahora es más específico
            print("❌ No hay datos de entrenamiento en la carpeta '.../train/smart'.")
            print("💡 Ejecuta primero las Opciones 3, 4 y 5.")
            return
        
        # Cargar modelo
        model = YOLO('yolov8n.pt')
        
        # Configuración automática
        training_config = {
            'data': str(self.project_root / "configs" / "dataset.yaml"),
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['training']['image_size'],
            'batch': self.config['training']['batch_size'],
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 0,
            'patience': 10,
            'save': True,
            'exist_ok': True,
            'project': str(self.project_root / "results"),
            'name': f"{work_type}_model",
            'amp': True,
        }
        
        print("⚙️ Configuración automática:")
        for key, value in training_config.items():
            print(f"   {key}: {value}")
        
        # Entrenar
        print("\n🎯 Iniciando entrenamiento...")
        try:
            results = model.train(**training_config)
            print("✅ ENTRENAMIENTO COMPLETADO!")
            return results
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
    
    def check_training_data(self):
        """
        Verifica que existan datos para entrenar
        EN LA CARPETA 'smart'
        """
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN! ---
        train_dir = self.project_root / "data" / "processed" / "images" / "train" / "smart"
        
        return train_dir.exists() and any(train_dir.glob("*.jpg"))

def main():
    trainer = UniversalTrainer()
    trainer.quick_train()

if __name__ == "__main__":
    main()