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
        'src/universal_trainer.py',
        'src/data_collector.py'
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
    for path in sys.path:
        print(f"   {path}")
    
    # Verificar imports
    print(f"\n📦 VERIFICANDO IMPORTS:")
    try:
        from src.setup_work import WorkConfigurator
        print("   ✅ src.setup_work")
    except Exception as e:
        print(f"   ❌ src.setup_work: {e}")
    
    try:
        from src.data_collector import DataCollector
        print("   ✅ src.data_collector")
    except Exception as e:
        print(f"   ❌ src.data_collector: {e}")
    
    try:
        from src.universal_trainer import UniversalTrainer
        print("   ✅ src.universal_trainer")
    except Exception as e:
        print(f"   ❌ src.universal_trainer: {e}")

def main():
    debug_environment()
    
    print("\n🎮 OPCIONES PRINCIPALES:")
    print("1. 🎯 Configurar nuevo tipo de trabajo")
    print("2. 🎥 Grabar datos de entrenamiento")
    print("3. 🏷️ Abrir etiquetador (LabelImg)")
    print("4. 🚀 Entrenar modelo")
    print("6. 🧠 Sistema avanzado de video")
    print("5. ❌ Salir")
    
    try:
        choice = input("\nSelecciona opción (1-5): ").strip()
        
        if choice == "1":
            print("🎯 Iniciando configuración...")
            from src.setup_work import main as setup_main
            setup_main()
            
        elif choice == "2":
            print("🎥 Iniciando grabación...")
            from src.data_collector import main as collector_main
            collector_main()
            
        elif choice == "3":
            print("🔧 Abriendo LabelImg...")
            os.system("labelImg")
            
        elif choice == "4":
            print("🚀 Iniciando entrenamiento...")
            from src.universal_trainer import main as trainer_main
            trainer_main()
            
        elif choice == "5":
            print("👋 ¡Hasta luego!")
        
        elif choice == "6":
            print("6. 🧠 Sistema avanzado de video")
            
        else:
            print("❌ Opción no válida")
            
    except Exception as e:
        print(f"💥 ERROR: {e}")
        input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main()