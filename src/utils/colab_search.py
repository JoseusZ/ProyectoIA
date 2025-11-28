"""
Entrenador Universal Híbrido (Local / Colab) v8 - FINAL
- Corrige el error de rutas relativas vs absolutas en Colab.
- Sincronizado perfectamente con colab_handler.py.
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
        # 1. Detectar entorno
        try:
            import google.colab
            self.IN_COLAB = True
            self.base_path = Path("/content") 
            print("☁️  Modo Nube (Colab) detectado.")
        except ImportError:
            self.IN_COLAB = False
            self.base_path = Path(__file__).parent.parent 
            print("💻 Modo Local detectado.")

        # 2. Definir raíz del proyecto
        if self.IN_COLAB:
            potential_root = self.base_path / "ProyectoIA"
            if potential_root.exists():
                self.project_root = potential_root
            else:
                self.project_root = self.base_path 
        else:
            self.project_root = self.base_path

        # 3. Cargar configuración
        self.config = self.load_config()
        if self.config is None:
            sys.exit(1)
            
        self.work_type = self.config.get('work_type', 'default_job')

    def load_config(self):
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            print(f"❌ Error: Configuración no encontrada en {config_path}")
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _update_dataset_yaml_for_colab(self):
        original_yaml = self.project_root / "configs" / "dataset.yaml"
        colab_yaml = self.project_root / "configs" / "dataset_colab.yaml"
        
        # --- CORRECCIÓN AQUÍ: Usar ruta relativa al proyecto ---
        # Antes buscaba en /content/data..., ahora busca en /content/ProyectoIA/data...
        colab_data_root = self.project_root / "data" / "processed"
        
        train_root = colab_data_root / "images" / "train"
        val_root = colab_data_root / "images" / "val"
        
        train_specific = train_root / self.work_type
        val_specific = val_root / self.work_type
        
        final_train = ""
        final_val = ""

        # Lógica de detección
        if train_specific.exists():
            print(f"✅ Train: Encontrado en subcarpeta '{self.work_type}'")
            final_train = str(train_specific)
        elif train_root.exists():
            print(f"✅ Train: Usando raíz estándar 'train'")
            final_train = str(train_root)
        else:
            print(f"❌ ERROR CRÍTICO: No encuentro datos de entrenamiento.")
            print(f"   Buscando en: {train_root}")
            print(f"   (Ruta base: {colab_data_root})")
            sys.exit(1)

        if val_specific.exists():
            print(f"✅ Val: Encontrado en subcarpeta '{self.work_type}'")
            final_val = str(val_specific)
        elif val_root.exists():
            print(f"✅ Val: Usando raíz estándar 'val'")
            final_val = str(val_root)
        else:
            print(f"⚠️  Advertencia: No encuentro carpeta de validación. Usando Train como Val.")
            final_val = final_train

        # Guardar YAML
        with open(original_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        data['path'] = str(colab_data_root)
        data['train'] = final_train
        data['val'] = final_val
        
        with open(colab_yaml, 'w') as f:
            yaml.dump(data, f)
            
        return str(colab_yaml)

    def _setup_drive_backup_callback(self, model):
        drive_backup_dir = Path("/content/drive/MyDrive/ProyectoIA/backup_weights") / self.work_type
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        
        def on_train_epoch_end(trainer):
            project = trainer.args.project
            name = trainer.args.name
            weights_dir = Path(project) / name / "weights"
            files = ['best.pt', 'last.pt']
            for f in files:
                local_f = weights_dir / f
                if local_f.exists():
                    try: shutil.copy2(local_f, drive_backup_dir / f)
                    except: pass 

        print(f"💾 Respaldo automático en Drive activado.")
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def run_training(self):
        print(f"🚀 ENTRENAMIENTO: {self.work_type.upper()}")
        print("=" * 50)

        if self.IN_COLAB:
            dataset_yaml_path = self._update_dataset_yaml_for_colab()
        else:
            dataset_yaml_path = str(self.project_root / "configs" / "dataset.yaml")

        t_params = self.config.get('training', {})
        if 'batch_size' in t_params: t_params['batch'] = t_params.pop('batch_size')
        if 'image_size' in t_params: t_params['imgsz'] = t_params.pop('image_size')

        resume_path = self.project_root / "results" / f"{self.work_type}_model" / "weights" / "best.pt"
        base_model = t_params.pop('base_model', 'yolov8n.pt')

        model_to_load = str(resume_path) if resume_path.exists() else base_model
        if resume_path.exists(): print(f"🔄 Retomando entrenamiento previo.")
        else: print(f"🆕 Iniciando desde cero.")

        final_args = {
            'data': dataset_yaml_path,
            'project': str(self.project_root / "results"),
            'name': f"{self.work_type}_model",
            'exist_ok': True,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'save': True,
            **t_params
        }
        
        model = YOLO(model_to_load)
        if self.IN_COLAB and os.path.exists("/content/drive"):
            self._setup_drive_backup_callback(model)

        print("\n⚙️  Iniciando YOLO...")
        try:
            model.train(**final_args)
            print("✅ ENTRENAMIENTO FINALIZADO")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if "CUDA" in str(e): print("💡 Reduce el batch_size.")

def main():
    UniversalTrainer().run_training()

if __name__ == "__main__":
    main()