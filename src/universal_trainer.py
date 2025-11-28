"""
Entrenador Universal Híbrido (Local / Colab) v10 - EL RESTAURADOR
- Busca automáticamente respaldos en Google Drive si no hay nada local.
- Recupera 'best.pt' o 'last.pt' para no perder progreso.
- Mantiene la protección de escritura y respaldo automático.
"""
import torch
import os
import sys
import yaml
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

class UniversalTrainer:
    def __init__(self):
        try:
            import google.colab
            self.IN_COLAB = True
            self.base_path = Path("/content") 
            print("☁️  Modo Nube (Colab) detectado.")
        except ImportError:
            self.IN_COLAB = False
            self.base_path = Path(__file__).parent.parent 
            print("💻 Modo Local detectado.")

        if self.IN_COLAB:
            potential_root = self.base_path / "ProyectoIA"
            if potential_root.exists():
                self.project_root = potential_root
            else:
                self.project_root = self.base_path 
        else:
            self.project_root = self.base_path

        self.config = self.load_config()
        if self.config is None: sys.exit(1)
        self.work_type = self.config.get('work_type', 'default_job')

    def load_config(self):
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            print(f"❌ Error: Configuración no encontrada en {config_path}")
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _verify_drive_access(self):
        """Verifica permisos de escritura en Drive"""
        drive_path = Path("/content/drive/MyDrive")
        if not drive_path.exists():
            # Intentar montar si no está montado
            print("⚠️ Drive no detectado. Intentando montar...")
            try:
                from google.colab import drive
                drive.mount('/content/drive')
            except:
                print("❌ ERROR CRÍTICO: Google Drive no accesible.")
                return False
            
        backup_dir = drive_path / "ProyectoIA" / "backup_weights" / self.work_type
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = backup_dir / "test_write_permission.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("Prueba.")
            os.remove(test_file)
            print(f"✅ ACCESO A DRIVE VERIFICADO: Escritura correcta.")
            return True
        except Exception as e:
            print(f"❌ ERROR: No tengo permisos de escritura en Drive. {e}")
            return False

    def _try_restore_from_drive(self):
        """
        Busca en Drive si existe un best.pt o last.pt y lo trae al entorno local.
        Retorna la ruta del archivo local si tuvo éxito, o None.
        """
        if not self.IN_COLAB: return None

        drive_dir = Path("/content/drive/MyDrive/ProyectoIA/backup_weights") / self.work_type
        local_dir = self.project_root / "results" / f"{self.work_type}_model" / "weights"
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔍 Buscando respaldos en Drive: {drive_dir} ...")

        # Prioridad 1: best.pt, Prioridad 2: last.pt
        for filename in ['best.pt', 'last.pt']:
            drive_file = drive_dir / filename
            if drive_file.exists():
                print(f"✨ ¡ENCONTRADO! {filename} detectado en Drive.")
                print(f"   📥 Descargando a entorno local para entrenar...")
                target_file = local_dir / filename
                try:
                    shutil.copy2(drive_file, target_file)
                    print(f"   ✅ Restauración completada: {target_file}")
                    return str(target_file)
                except Exception as e:
                    print(f"   ❌ Error al copiar desde Drive: {e}")
        
        print("   ⚠️ No se encontraron respaldos previos en Drive. Se iniciará desde cero.")
        return None

    def _setup_drive_backup_callback(self, model):
        if not self._verify_drive_access():
            print("🛑 DETENIENDO: Sin respaldo seguro no arranco.")
            sys.exit(1)

        drive_backup_dir = Path("/content/drive/MyDrive/ProyectoIA/backup_weights") / self.work_type
        
        def on_train_epoch_end(trainer):
            project = trainer.args.project
            name = trainer.args.name
            weights_dir = Path(project) / name / "weights"
            
            for filename in ['best.pt', 'last.pt']:
                local_file = weights_dir / filename
                drive_file = drive_backup_dir / filename
                
                if local_file.exists():
                    try:
                        shutil.copy2(local_file, drive_file)
                        current_time = time.strftime("%H:%M")
                        # Mensaje minimalista para no saturar
                        print(f"   ☁️ [{current_time}] Backup OK: {filename}")
                    except Exception:
                        pass # Silencioso si falla uno, ya avisamos al inicio
                
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def _update_dataset_yaml_for_colab(self):
        original_yaml = self.project_root / "configs" / "dataset.yaml"
        colab_yaml = self.project_root / "configs" / "dataset_colab.yaml"
        colab_data_root = self.project_root / "data" / "processed"
        
        train_root = colab_data_root / "images" / "train"
        val_root = colab_data_root / "images" / "val"
        train_specific = train_root / self.work_type
        val_specific = val_root / self.work_type
        
        final_train, final_val = "", ""

        if train_specific.exists(): final_train = str(train_specific)
        elif train_root.exists(): final_train = str(train_root)
        else:
            print(f"❌ ERROR DATOS: No encuentro {train_root}")
            sys.exit(1)

        if val_specific.exists(): final_val = str(val_specific)
        elif val_root.exists(): final_val = str(val_root)
        else: final_val = final_train

        with open(original_yaml, 'r') as f: data = yaml.safe_load(f)
        data['path'] = str(colab_data_root)
        data['train'] = final_train
        data['val'] = final_val
        
        with open(colab_yaml, 'w') as f: yaml.dump(data, f)
        return str(colab_yaml)

    def run_training(self):
        print(f"🚀 ENTRENAMIENTO v10 (RESTAURADOR): {self.work_type.upper()}")
        print("=" * 50)

        if self.IN_COLAB:
            dataset_yaml_path = self._update_dataset_yaml_for_colab()
        else:
            dataset_yaml_path = str(self.project_root / "configs" / "dataset.yaml")

        t_params = self.config.get('training', {})
        if 'batch_size' in t_params: t_params['batch'] = t_params.pop('batch_size')
        if 'image_size' in t_params: t_params['imgsz'] = t_params.pop('image_size')

        # --- LÓGICA DE RECUPERACIÓN INTELIGENTE ---
        local_resume_path = self.project_root / "results" / f"{self.work_type}_model" / "weights" / "best.pt"
        base_model = t_params.pop('base_model', 'yolov8n.pt')
        model_to_load = base_model

        # 1. Buscamos Localmente
        if local_resume_path.exists():
            print(f"🔄 Usando modelo LOCAL encontrado: {local_resume_path}")
            model_to_load = str(local_resume_path)
        # 2. Si no hay local y estamos en Colab, buscamos en Drive
        elif self.IN_COLAB:
            restored_model = self._try_restore_from_drive()
            if restored_model:
                model_to_load = restored_model
                print("🔄 Retomando entrenamiento desde respaldo de Drive.")
            else:
                print(f"🆕 Iniciando entrenamiento desde cero: {base_model}")
        else:
            print(f"🆕 Iniciando entrenamiento desde cero (Local): {base_model}")

        # Configuración final
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
        
        if self.IN_COLAB:
            self._setup_drive_backup_callback(model)

        print("\n⚙️  Iniciando motor YOLO...")
        try:
            model.train(**final_args)
            print("✅ ENTRENAMIENTO FINALIZADO")
        except Exception as e:
            print(f"\n❌ Error en entrenamiento: {e}")

def main():
    UniversalTrainer().run_training()

if __name__ == "__main__":
    main()