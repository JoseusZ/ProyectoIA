"""
INSTALADOR DE ENTORNO GLOBAL (v1.1)
Este script prepara el entorno de Python para el proyecto 
de análisis de productividad.

Requiere ser ejecutado con Python 3.10.

Detecta automáticamente el SO, la presencia de GPU (CUDA)
y maneja las dependencias problemáticas (polars).
"""

import os
import sys
import platform
import subprocess
import venv
import shutil  # <-- Importante para borrar la carpeta
from pathlib import Path

# --- Configuración de Dependencias ---
COMMON_LIBRARIES = [
    "ultralytics",
    "mediapipe",
    "opencv-python",
    "PyYAML",
    "scikit-learn",
    "matplotlib",
    "lap",
    "polars[rtcompat]"  # ¡La clave! Evita errores de AVX2
]
# -------------------------------------

class EnvironmentSetup:
    """
    Clase que maneja la lógica de detección e instalación 
    del entorno.
    """
    def __init__(self):
        self.project_root = Path(__file__).parent.resolve()
        self.venv_dir = self.project_root / ".venv"
        
        self.os_name = platform.system().lower()
        self.arch = platform.machine().lower()
        
        if self.os_name == "windows":
            self.is_windows = True
            # Usamos el python que está ejecutando este script
            self.python_exec = sys.executable
            self.pip_exec = self.venv_dir / "Scripts" / "pip.exe"
            self.activate_msg = f"Activa el entorno con: .\\.venv\\Scripts\\Activate.ps1"
        else:
            self.is_windows = False
            self.python_exec = sys.executable
            self.pip_exec = self.venv_dir / "bin" / "pip"
            self.activate_msg = f"Activa el entorno con: source .venv/bin/activate"
            
        print(f"Sistema Detectado: {self.os_name} ({self.arch})")
        print(f"Raíz del Proyecto: {self.project_root}")
        print(f"Usando Python: {platform.python_version()} (Desde: {self.python_exec})")


    def run_command(self, command, description):
        """Ejecuta un comando de terminal de forma robusta."""
        print(f"\n⏳ {description}...")
        try:
            subprocess.run(command, check=True, shell=self.is_windows)
            print(f"✅ ¡Éxito! {description.split('...')[0]} completado.")
        except subprocess.CalledProcessError as e:
            print(f"\n" + "="*50)
            print(f"❌ ¡ERROR FATAL durante: {description}!")
            print(f"   Comando que falló: {' '.join(command)}")
            print(f"   Error: {e}")
            print("El script se detendrá. Revisa el error e inténtalo de nuevo.")
            print("="*50)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"\n" + "="*50)
            print(f"❌ ¡ERROR FATAL: Comando no encontrado!")
            print(f"   Comando: {command[0]}")
            print(f"   Error: {e}")
            print("   ¿Está el programa (ej. 'python') en tu PATH del sistema?")
            print("="*50)
            sys.exit(1)

    def create_venv(self):
        """Crea el entorno virtual."""
        # Esta función ahora solo se llama si .venv no existe o fue borrado.
        self.run_command(
            [self.python_exec, "-m", "venv", str(self.venv_dir)],
            "Creando nuevo entorno virtual (.venv)"
        )
            
    def detect_cuda(self):
        """Intenta detectar una GPU NVIDIA usando 'nvidia-smi'."""
        if self.is_windows:
            command = ["nvidia-smi"]
        else:
            command = ["which", "nvidia-smi"]
            
        print("\n🔍 Detectando GPU NVIDIA (CUDA)...")
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=self.is_windows)
            print("✅ ¡GPU NVIDIA detectada! Se instalará PyTorch con soporte CUDA.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("INFO: No se detectó GPU NVIDIA (nvidia-smi no encontrado o falló).")
            print("      Se instalará PyTorch en modo 'solo CPU'.")
            return False

    def install_dependencies(self):
        """Instala todas las librerías necesarias."""
        
        # 1. Actualizar pip
        self.run_command(
            [str(self.pip_exec), "install", "--upgrade", "pip", "setuptools", "wheel"],
            "Actualizando pip y setuptools"
        )
        
        # 2. Instalar PyTorch (INTELIGENTE)
        has_cuda = self.detect_cuda()
        
        if has_cuda:
            torch_command = [str(self.pip_exec), "install", "torch", "torchvision", "torchaudio"]
            desc = "Instalando PyTorch (con CUDA)"
        else:
            torch_command = [
                str(self.pip_exec), "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            desc = "Instalando PyTorch (solo CPU)"
            
        self.run_command(torch_command, desc)
        
        # 3. Instalar el resto de librerías
        self.run_command(
            [str(self.pip_exec), "install"] + COMMON_LIBRARIES,
            f"Instalando {len(COMMON_LIBRARIES)} librerías del proyecto"
        )

    def run_setup(self):
        """Ejecuta todo el proceso de instalación."""
        print("--- INICIANDO INSTALADOR (usando Python " + platform.python_version() + ") ---")
        
        # --- 2. Manejo de .venv existente (LÓGICA MEJORADA) ---
        venv_ready = False
        if self.venv_dir.exists():
            print("\nINFO: La carpeta '.venv' ya existe.")
            print("      Puede estar incompleta, rota, o ser de otra versión de Python.")
            
            while True:
                choice = input("¿Qué quieres hacer?\n"
                               "  [1] Borrarla y crearla de nuevo (Recomendado)\n"
                               "  [2] Intentar re-usarla (Riesgoso)\n"
                               "  [3] Salir\n"
                               "Opción: ").strip()
                
                if choice == '1':
                    print("\nBorrando .venv existente...")
                    try:
                        shutil.rmtree(self.venv_dir)
                    except OSError as e:
                        print(f"❌ Error al borrar .venv: {e}")
                        print("   Por favor, borra la carpeta '.venv' manualmente y vuelve a ejecutar.")
                        sys.exit(1)
                    
                    self.create_venv() # Crear una nueva
                    venv_ready = True
                    break
                elif choice == '2':
                    print("INFO: Intentando re-usar .venv...")
                    if not self.pip_exec.exists():
                        print(f"❌ ¡ERROR! El .venv existente está roto!")
                        print(f"   No se encuentra: {self.pip_exec}")
                        print(f"   Por favor, borra la carpeta '.venv' (Opción 1) y vuelve a ejecutar.")
                        # No salimos, volvemos al bucle
                    else:
                        print("   .venv parece válido. Continuando...")
                        venv_ready = True
                        break
                elif choice == '3':
                    print("Saliendo. No se ha instalado nada.")
                    sys.exit(0)
                else:
                    print("Opción no válida. Intenta de nuevo.")
        
        else:
            self.create_venv() # No existía, crearla
            venv_ready = True

        # --- 3. Instalar Dependencias ---
        if venv_ready:
            self.install_dependencies()
        
            print("\n" + "="*50)
            print("🎉 ¡INSTALACIÓN COMPLETADA!")
            print("Todos los paquetes se instalaron correctamente.")
            print(f"\nPRÓXIMO PASO:")
            print(f"En tu terminal, ejecuta lo siguiente para activar el entorno:")
            print(f"   {self.activate_msg}")
            print(f"Luego, podrás correr tu proyecto:")
            print(f"   python run_project.py")
            print("="*50)

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    
    # --- 1. VERIFICACIÓN DE VERSIÓN DE PYTHON ---
    TARGET_MAJOR, TARGET_MINOR = 3, 10
    current_ver = sys.version_info
    
    if (current_ver.major != TARGET_MAJOR or current_ver.minor != TARGET_MINOR):
        
        print("="*60)
        print(f"❌ ERROR: VERSIÓN DE PYTHON INCORRECTA")
        print(f"   Se esperaba Python {TARGET_MAJOR}.{TARGET_MINOR}, pero estás usando {current_ver.major}.{current_ver.minor}")
        print("   Este script debe ejecutarse con la versión 3.10 para asegurar compatibilidad.")
        print("   El script se detendrá.")
        print("\n--- CÓMO ARREGLARLO ---")
        
        os_name = platform.system().lower()
        if os_name == "linux":
            print("Estás en Linux. Sigue estos pasos en tu terminal:")
            print("   1. (Si no lo has hecho) Instala Python 3.10:")
            print("      sudo apt update")
            print("      sudo apt install software-properties-common")
            print("      sudo add-apt-repository ppa:deadsnakes/ppa")
            print("      sudo apt install python3.10 python3.10-venv")
            print("\n   2. Ejecuta este script de nuevo USANDO python3.10:")
            print(f"      python3.10 {Path(__file__).name}")
            
        elif os_name == "windows":
            print("Estás en Windows. Por favor:")
            print("   1. Ve a 'Agregar o quitar programas' y desinstala esta versión de Python.")
            print("   2. Ve a https://www.python.org/downloads/ y descarga un instalador de Python 3.10 (ej. 3.10.11).")
            print("   3. Durante la instalación, ASEGÚRATE de marcar la casilla 'Add Python to PATH'.")
            print("   4. Abre una NUEVA terminal y ejecuta de nuevo este script:")
            print(f"      python {Path(__file__).name}")
        
        else: # macOS, etc.
            print(f"Estás en {os_name}. Por favor, instala Python 3.10 (ej. via brew) y")
            print(f"asegúrate de que sea la versión por defecto o ejecuta el script con 'python3.10'.")
            
        sys.exit(1) # Detener el script
    
    # --- Si la versión es correcta (3.10), continuar ---
    setup_manager = EnvironmentSetup()
    setup_manager.run_setup()