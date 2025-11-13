"""
Script de configuración y verificación del entorno de desarrollo
"""
import os
import sys
import torch
import cv2
from pathlib import Path

class DevelopmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_paths()
        
    def setup_paths(self):
        """Configura los paths del proyecto"""
        # Agregar src al path de Python
        src_path = self.project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.append(str(src_path))
        
        print(f"📁 Project Root: {self.project_root}")
        print(f"📁 SRC Path: {src_path}")
    
    def check_environment(self):
        """Verifica el entorno de desarrollo"""
        print("🔍 VERIFICANDO ENTORNO DE DESARROLLO")
        print("=" * 50)
        
        # 1. Verificar hardware
        print("1. HARDWARE:")
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 2. Verificar librerías
        print("\n2. LIBRERÍAS:")
        print(f"   ✅ OpenCV: {cv2.__version__}")
        
        try:
            from ultralytics import YOLO
            print("   ✅ Ultralytics YOLOv8")
        except ImportError as e:
            print(f"   ❌ Ultralytics: {e}")
        
        # 3. Verificar estructura de proyecto
        print("\n3. ESTRUCTURA DEL PROYECTO:")
        required_dirs = [
            "data/raw/videos",
            "data/processed/images/train",
            "data/processed/images/val", 
            "src/utils",
            "configs",
            "models",
            "notebooks"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print(f"   ✅ {dir_path}")
            else:
                print(f"   ❌ {dir_path}")
        
        print("=" * 50)
        print("🎉 ENTORNO DE DESARROLLO LISTO!")
    
    def test_yolov8_inference(self):
        """Prueba básica de inferencia con YOLOv8"""
        print("\n🧪 TEST DE INFERENCIA YOLOv8:")
        try:
            from ultralytics import YOLO
            
            # Cargar modelo nano (más liviano)
            model = YOLO('yolov8n.pt')
            
            # Crear imagen dummy para prueba
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Realizar inferencia
            results = model(dummy_image, verbose=False)
            
            print("   ✅ Inferencia básica funcionando")
            print(f"   ✅ Objetos detectados: {len(results[0].boxes) if results[0].boxes else 0}")
            
        except Exception as e:
            print(f"   ❌ Error en inferencia: {e}")

if __name__ == "__main__":
    dev = DevelopmentSetup()
    dev.check_environment()
    dev.test_yolov8_inference()