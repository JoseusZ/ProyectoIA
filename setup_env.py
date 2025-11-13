"""
INSTALADOR DE ENTORNO GLOBAL (v1.0)
Este script prepara el entorno de Python para el proyecto 
de análisis de productividad.

Detecta automáticamente el SO, la presencia de GPU (CUDA)
y maneja las dependencias problemáticas (polars).

Para ejecutarlo:
1. Asegúrate de tener Python 3.8+ instalado.
2. Coloca este script en la raíz de tu carpeta de proyecto.
3. Abre una terminal y ejecuta: python setup_env.py
"""

import os
import sys
import platform
import subprocess
import venv
from pathlib import Path

# --- Configuración de Dependencias ---
# Estas son las librerías que tu proyecto necesita.
# PyTorch se instala por separado.
COMMON_LIBRARIES = [
    "ultralytics",
    "mediapipe",
    "opencv-python",  # Para cv2.imshow()
    "PyYAML",
    "scikit-learn",
    "matplotlib",
    "lap",
    "polars[rtcompat]"  # ¡La clave! Instala la versión compatible para evitar errores de AVX2
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
        
        # Detectar Sistema Operativo y Arquitectura
        self.os_name = platform.system().lower()
        self.arch = platform.machine().lower()
        
        # Determinar nombres de ejecutables
        if self.os_name == "windows":
            self.is_windows = True
            self.python_exec = sys.executable
            self.pip_exec = self.venv_dir / "Scripts" / "pip.exe"
            self.activate_msg = f"Activa el entorno con: .\\.venv\\Scripts\\Activate.ps1"
        else:
            self.is_windows = False
            self.python_exec = sys.executable  # Usa el python que ejecuta este script
            self.pip_exec = self.venv_dir / "bin" / "pip"
            self.activate_msg = f"Activa el entorno con: source .venv/bin/activate"
            
        print(f"Sistema Detectado: {self.os_name} ({self.arch})")
        print(f"Raíz del Proyecto: {self.project_root}")

    def run_command(self, command, description):
        """Ejecuta un comando de terminal de forma robusta."""
        print(f"\n⏳ {description}...")
        try:
            # shell=True es necesario en Windows para algunos comandos
            subprocess.run(command, check=True, shell=self.is_windows)
            print(f"✅ ¡Éxito! {description.split('...')[0]} completado.")
        except subprocess.CalledProcessError as e:
            print(f"\n" + "="*50)
            print(f"❌ ¡ERROR FATAL durante: {description}!")
            print(f"   Comando que falló: {' '.join(command)}")
            print(f"   Error: {e}")
            print("El script se detendrá. Revisa el error e inténtalo de nuevo.")
            print("="*50)
            sys.exit(1) # Detener el script si un paso falla
        except FileNotFoundError as e:
            print(f"\n" + "="*50)
            print(f"❌ ¡ERROR FATAL: Comando no encontrado!")
            print(f"   Comando: {command[0]}")
            print(f"   Error: {e}")
            print("   ¿Está el programa (ej. 'python') en tu PATH del sistema?")
            print("="*50)
            sys.exit(1)

    def create_venv(self):
        """Crea el entorno virtual si no existe."""
        if self.venv_dir.exists():
            print(f"INFO: El entorno virtual '.venv' ya existe. Omitiendo creación.")
        else:
            # Usamos el módulo venv de la librería estándar
            self.run_command(
                [self.python_exec, "-m", "venv", str(self.venv_dir)],
                "Creando entorno virtual (.venv)"
            )
            
    def detect_cuda(self):
        """
        Intenta detectar una GPU NVIDIA usando 'nvidia-smi'.
        Este es el método más fiable antes de instalar PyTorch.
        """
        if self.is_windows:
            command = ["nvidia-smi"]
        else:
            # En Linux, 'which' puede encontrar el comando
            command = ["which", "nvidia-smi"]
            
        print("\n🔍 Detectando GPU NVIDIA (CUDA)...")
        try:
            # Usamos Popen para suprimir la salida masiva de nvidia-smi
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
            # Comando estándar para instalar PyTorch con CUDA
            torch_command = [str(self.pip_exec), "install", "torch", "torchvision", "torchaudio"]
            desc = "Instalando PyTorch (con CUDA)"
        else:
            # Comando especial para instalar PyTorch (solo CPU)
            # Esto es más rápido y compatible si no hay GPU.
            torch_command = [
                str(self.pip_exec), "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            desc = "Instalando PyTorch (solo CPU)"
            
        self.run_command(torch_command, desc)
        
        # 3. Instalar el resto de librerías (incluyendo polars[rtcompat])
        self.run_command(
            [str(self.pip_exec), "install"] + COMMON_LIBRARIES,
            f"Instalando {len(COMMON_LIBRARIES)} librerías del proyecto"
        )

    def run_setup(self):
        """Ejecuta todo el proceso de instalación."""
        print("--- INICIANDO INSTALADOR DE ENTORNO GLOBAL ---")
        
        self.create_venv()
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
    if sys.version_info < (3, 8):
        print("❌ ERROR: Este script requiere Python 3.8 o superior.")
        print("   Tu versión es: {platform.python_version()}")
        sys.exit(1)
        
    setup_manager = EnvironmentSetup()
    setup_manager.run_setup()