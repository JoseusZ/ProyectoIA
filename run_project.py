"""
VERSIÓN DIAGNÓSTICO - Sistema de Análisis de Productividad
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
        'src/utils/merge_tool.py', # <-- ¡AÑADIDO!
        'src/utils/productivity_monitor.py',
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
    debug_environment()
    
    print("\n🎮 OPCIONES PRINCIPALES DEL PROYECTO:")
    print("--- FASE 1: PREPARACIÓN ---")
    print("1. 🎯 Configurar nuevo tipo de trabajo")
    print("2. 🎥 Grabar datos de entrenamiento")
    print("3. 🧠 Extraer Frames (Procesador de Video)")
    print("--- FASE 2: ETIQUETADO ---")
    print("4. 🤖 Pre-etiquetar imágenes (Automático)")
    print("5. 🏷️  Corregir etiquetas (Manual)")
    print("6. 🛠️  Fusionar Dataset (Añadir datos)") # <-- ¡NUEVO!
    print("--- FASE 3: ENTRENAMIENTO Y EJECUCIÓN ---")
    print("7. 🚀 Entrenar modelo") # <-- Re-numerado (era 6)
    print("8. 🕵️  Iniciar Monitor de Productividad (YOLO + MediaPipe)") # <-- Re-numerado (era 7)
    print("9. ❌ Salir") # <-- Re-numerado (era 8)
    
    try:
        choice = input("\nSelecciona opción (1-9): ").strip() # <-- Rango actualizado
        
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

        # --- BLOQUE NUEVO ---
        elif choice == "6":
            print("🛠️  Iniciando herramienta de fusión de datasets...")
            from src.utils.merge_tool import main as merge_main
            merge_main()
        # --- FIN DE BLOQUE NUEVO ---

        elif choice == "7": # <-- Re-numerado (era 6)
            print("🚀 Iniciando entrenamiento...")
            from src.universal_trainer import main as trainer_main
            trainer_main()
            
        elif choice == "8": # <-- Re-numerado (era 7)
            print("🕵️  Iniciando monitor de productividad...")
            from src.utils.productivity_monitor import main as monitor_main
            monitor_main()
        
        elif choice == "9": # <-- Re-numerado (era 8)
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