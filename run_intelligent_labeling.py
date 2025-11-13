"""
Ejecutor independiente del sistema inteligente de etiquetado
"""
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

if __name__ == "__main__":
    try:
        from src.utils.intelligent_labeling import main
        main()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que:")
        print("   1. El archivo intelligent_labeling.py esté en src/utils/")
        print("   2. Estés ejecutando desde la raíz del proyecto")