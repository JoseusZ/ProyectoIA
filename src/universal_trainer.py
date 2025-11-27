"""
Entrenador Universal Híbrido (Local / Colab) v5
- Detecta automáticamente el entorno.
- Ajusta rutas de dataset dinámicamente.
- Implementa respaldo automático a Drive si está en Colab.
- Mantiene la memoria (Transfer Learning) si existe un modelo previo.
"""
import torch
import os
import sys
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

class UniversalTrainer:
    def __init__(self):
        # Detectar si estamos en Colab
        try:
            import google.colab
            self.IN_COLAB = True
            self.base_path = Path("/content") # Ruta raíz en Colab
            print("☁️  Modo Nube (Colab) detectado.")
        except ImportError:
            self.IN_COLAB = False
            self.base_path = Path(__file__).parent.parent # Ruta raíz Local
            print("💻 Modo Local detectado.")

        # En Colab, el proyecto clonado suele estar en /content/ProyectoIA
        # Ajustamos self.project_root según el entorno para encontrar configs
        if self.IN_COLAB:
            # Asumimos que el repo se clonó en /content/ProyectoIA
            potential_root = self.base_path / "ProyectoIA"
            if potential_root.exists():
                self.project_root = potential_root
            else:
                self.project_root = self.base_path # Fallback
        else:
            self.project_root = self.base_path

        self.config = self.load_config()
        if self.config is None:
            sys.exit(1)
            
        self.work_type = self.config.get('work_type', 'default_job')

    def load_config(self):
        """Carga la configuración del trabajo"""
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            print(f"❌ Error: Configuración no encontrada en {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _update_dataset_yaml_for_colab(self):
        """
        Crea una copia temporal del dataset.yaml con las rutas ajustadas para Colab.
        """
        original_yaml = self.project_root / "configs" / "dataset.yaml"
        colab_yaml = self.project_root / "configs" / "dataset_colab.yaml"
        
        # Rutas donde Colab Handler pone los datos
        # Nota: Ajusta esto si tu handler usa otra ruta (ej. /content/data/processed)
        colab_data_root = "/content/data/processed" 
        
        with open(original_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        # Sobrescribir rutas con las de Colab
        data['path'] = colab_data_root # Raíz opcional
        data['train'] = f"{colab_data_root}/images/train/{self.work_type}"
        data['val'] = f"{colab_data_root}/images/val/{self.work_type}"
        
        with open(colab_yaml, 'w') as f:
            yaml.dump(data, f)
            
        print(f"🔄 Rutas de dataset actualizadas para Colab en: {colab_yaml}")
        return str(colab_yaml)

    def _setup_drive_backup_callback(self, model):
        """
        Configura un callback para guardar best.pt en Drive en cada época.
        """
        drive_backup_dir = Path("/content/drive/MyDrive/ProyectoIA/backup_weights") / self.work_type
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        
        def on_train_epoch_end(trainer):
            # Ruta local donde YOLO guarda
            project = trainer.args.project
            name = trainer.args.name
            weights_dir = Path(project) / name / "weights"
            
            files = ['best.pt', 'last.pt']
            for f in files:
                local_f = weights_dir / f
                if local_f.exists():
                    try:
                        shutil.copy2(local_f, drive_backup_dir / f)
                    except Exception as e:
                        print(f"⚠️  No se pudo respaldar en Drive: {e}")

        print(f"💾 Respaldo automático en Drive activado: {drive_backup_dir}")
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def run_training(self):
        print(f"🚀 ENTRENAMIENTO PARA: {self.work_type.upper()}")
        print("=" * 50)

        # 1. Preparar Dataset YAML
        if self.IN_COLAB:
            # En Colab usamos el YAML modificado
            dataset_yaml_path = self._update_dataset_yaml_for_colab()
        else:
            # En Local usamos el original
            dataset_yaml_path = str(self.project_root / "configs" / "dataset.yaml")
            # Verificación de datos solo en local (en Colab confiamos en el handler)
            if not self.check_training_data_local():
                print("❌ Datos locales no encontrados. Revisa tus carpetas.")
                return

        # 2. Configuración de Entrenamiento
        training_params = self.config.get('training', {})
        
        # Normalización de claves
        if 'batch_size' in training_params: training_params['batch'] = training_params.pop('batch_size')
        if 'image_size' in training_params: training_params['imgsz'] = training_params.pop('image_size')

        # 3. Lógica de Memoria (Transfer Learning)
        # Buscamos modelo previo en la carpeta de resultados del proyecto
        resume_model_path = self.project_root / "results" / f"{self.work_type}_model" / "weights" / "best.pt"
        base_model_config = training_params.pop('base_model', 'yolov8n.pt')

        model_to_load = base_model_config
        if resume_model_path.exists():
            print(f"✅ Modelo previo detectado: {resume_model_path}")
            model_to_load = str(resume_model_path)
        else:
            print(f"🆕 Iniciando entrenamiento desde cero con: {base_model_config}")

        # 4. Parámetros estáticos
        static_params = {
            'data': dataset_yaml_path,
            'project': str(self.project_root / "results"),
            'name': f"{self.work_type}_model",
            'exist_ok': True,
        }

        # 5. Defaults
        device = 0 if torch.cuda.is_available() else 'cpu'
        training_params.setdefault('device', device)
        training_params.setdefault('workers', 2 if self.IN_COLAB else 0) # Colab tiene más CPU
        training_params.setdefault('patience', 10)
        training_params.setdefault('save', True)
        
        final_config = {**training_params, **static_params}

        # 6. Cargar Modelo
        model = YOLO(model_to_load)

        # 7. Activar Respaldo a Drive (Solo Colab)
        if self.IN_COLAB and os.path.exists("/content/drive"):
            self._setup_drive_backup_callback(model)

        # 8. Ejecutar
        print("\n⚙️  Iniciando motor YOLO...")
        try:
            results = model.train(**final_config)
            print("✅ ENTRENAMIENTO EXITOSO")
            return results
        except Exception as e:
            print(f"\n❌ Error Crítico: {e}")
            print("Consejo: Si es error de memoria (CUDA OOM), reduce el 'batch' en work_config.yaml")

    def check_training_data_local(self):
        """Verificación simple para entorno local"""
        labels_dir = self.project_root / "data" / "processed" / "labels" / "train" / self.work_type
        return labels_dir.exists() and any(labels_dir.glob("*.txt"))

def main():
    trainer = UniversalTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()