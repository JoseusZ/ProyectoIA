"""
VERSIÓN DIAGNÓSTICO - Sistema de Análisis de Productividad
Actualizado con Integración Google Colab
"""
import os
import sys
from pathlib import Path

def debug_environment():
    print("🐛 MODO DIAGNÓSTICO ACTIVADO")
    print("=" * 50)
    
    # Verificar directorio actual
    current_dir = Path().absolute()
    print(f"📁 Directorio actual: {current_dir}")
    
    # Verificar estructura
    essential_files = [
        'run_project.py',
        'src/__init__.py',
        'src/setup_work.py', 
        'src/data_collector.py',
        'src/utils/advanced_video_processor.py',
        'src/utils/auto_etiquetador.py',
        'src/utils/intelligent_labeling.py',
        'src/utils/merge_tool.py',
        'src/utils/productivity_monitor.py',
        'src/utils/colab_handler.py', # <-- ¡AÑADIDO PARA VERIFICACIÓN!
        'src/universal_trainer.py',
    ]
    
    print("\n🔍 VERIFICANDO ARCHIVOS:")
    for file in essential_files:
        file_path = current_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - FALTANTE")
    
    # Verificar Python path
    print(f"\n🐍 Python Path:")
    # Añadir 'src' al path si no está
    src_path = str(current_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        
    # Añadir la raíz del proyecto al path
    if str(current_dir) not in sys.path:
        sys.path.insert(1, str(current_dir))
        
    for path in sys.path:
        print(f"   {path}")

def main():
    # 1. Configurar entorno y rutas
    debug_environment()

    # --- INICIO INTEGRACIÓN GOOGLE COLAB ---
    try:
        # Intentamos importar el handler. Como debug_environment ya añadió 'src' al path,
        # podemos importar desde src.utils o utils dependiendo de cómo lo resuelva Python.
        print("\n☁️  Verificando entorno de ejecución...")
        try:
            from src.utils.colab_handler import check_and_setup_colab
        except ImportError:
            # Intento alternativo por si src ya es root en algunos entornos
            from utils.colab_handler import check_and_setup_colab
            
        # Ejecutar la comprobación
        base_path = str(Path().absolute())
        check_and_setup_colab(base_path)
    except ImportError:
        print("⚠️  No se encontró 'src/utils/colab_handler.py'. Saltando chequeo de nube.")
    except Exception as e:
        # Si falla (ej. no estamos en Colab y no tiene dependencias), solo continuamos
        # print(f"ℹ️  Info entorno: {e}") 
        pass
    # --- FIN INTEGRACIÓN GOOGLE COLAB ---
    
    print("\n🎮 OPCIONES PRINCIPALES DEL PROYECTO:")
    print("--- FASE 1: PREPARACIÓN ---")
    print("1. 🎯 Configurar nuevo tipo de trabajo")
    print("2. 🎥 Grabar datos de entrenamiento")
    print("3. 🧠 Extraer Frames (Procesador de Video)")
    print("--- FASE 2: ETIQUETADO ---")
    print("4. 🤖 Pre-etiquetar imágenes (Automático)")
    print("5. 🏷️  Corregir etiquetas (Manual)")
    print("6. 🛠️  Fusionar Dataset (Añadir datos)")
    print("--- FASE 3: ENTRENAMIENTO Y EJECUCIÓN ---")
    print("7. 🚀 Entrenar modelo")
    print("8. 🕵️  Iniciar Monitor de Productividad (YOLO + MediaPipe)")
    print("9. ❌ Salir")
    
    try:
        choice = input("\nSelecciona opción (1-9): ").strip()
        
        if choice == "1":
            print("🎯 Iniciando configuración...")
            from src.setup_work import main as setup_main
            setup_main()
            
        elif choice == "2":
            print("🎥 Iniciando grabación...")
            from src.data_collector import main as collector_main
            collector_main()
            
        elif choice == "3":
            print("🧠 Iniciando sistema avanzado de video...")
            from src.utils.advanced_video_processor import main as video_processor_main
            video_processor_main()
            
        elif choice == "4":
            print("🤖 Iniciando auto-etiquetado...")
            from src.utils.auto_etiquetador import main as auto_label_main
            auto_label_main()

        elif choice == "5":
            print("🏷️  Iniciando herramienta de corrección...")
            from src.utils.intelligent_labeling import main as labeling_main
            labeling_main()

        elif choice == "6":
            print("🛠️  Iniciando herramienta de fusión de datasets...")
            from src.utils.merge_tool import main as merge_main
            merge_main()

        elif choice == "7":
            print("🚀 Iniciando entrenamiento...")
            from src.universal_trainer import main as trainer_main
            trainer_main()
            
        elif choice == "8":
            print("🕵️  Iniciando monitor de productividad...")
            from src.utils.productivity_monitor import main as monitor_main
            monitor_main()
        
        elif choice == "9":
            print("👋 ¡Hasta luego!")
            
        else:
            print("❌ Opción no válida")
            
    except ImportError as e:
        print(f"\n💥 ERROR DE IMPORTACIÓN: {e}")
        print("💡 Asegúrate de que el archivo existe y que no hay errores de sintaxis.")
    except Exception as e:
        print(f"\n💥 ERROR INESPERADO: {e}")
        print("El programa se cerrará.")
        input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main()