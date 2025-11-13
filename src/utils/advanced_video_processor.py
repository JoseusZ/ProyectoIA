"""
SISTEMA AVANZADO DE PROCESAMIENTO DE VIDEO
Múltiples estrategias de extracción + Análisis inteligente
"""
import yaml
import sys
import cv2
import os
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class AdvancedVideoProcessor:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.config = self._load_config() # Cargar config
        if self.config is None:
            sys.exit(1) # Salir si no hay config
        
        self.work_type = self.config.get('work_type', 'default_job')
        self.setup_directories() # Ahora setup_directories() puede usar self.work_type
        
    def _load_config(self):
        """Carga el work_config.yaml para saber dónde guardar los archivos"""
        config_path = self.project_root / "configs" / "work_config.yaml"
        if not config_path.exists():
            print("❌ Error: 'configs/work_config.yaml' no encontrado.")
            print("💡 Ejecuta la 'Opción 1: Configurar...' primero.")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Crea la estructura avanzada de directorios (v2 - Global)"""
        # Directorios estáticos
        directories = [
            "data/raw/videos",
            "data/analysis/reports",
            "data/analysis/visualizations",
            "results/processing_logs"
        ]
        
        # Directorios dinámicos (basados en el work_type)
        # ¡Esta es la estructura que YOLO espera!
        dynamic_dirs = [
            f"data/processed/images/train/{self.work_type}",
            f"data/processed/images/val/{self.work_type}", # Aunque no la usemos, la creamos
            f"data/processed/labels/train/{self.work_type}", # ¡Importante!
            f"data/processed/labels/val/{self.work_type}",   # ¡Importante!
        ]
        
        for directory in (directories + dynamic_dirs):
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
    
    def analyze_video_content(self, video_path):
        """
        Analiza el contenido del video para estrategia óptima
        """
        print(f"🔍 ANALIZANDO VIDEO: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Métricas del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Muestreo para análisis
        sample_frames = min(100, total_frames)
        frame_interval = max(1, total_frames // sample_frames)
        
        motion_levels = []
        brightness_levels = []
        
        ret, prev_frame = cap.read()
        if not ret:
            return None
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(sample_frames):
            frame_pos = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Análisis de movimiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_gray, gray)
            motion_pixels = cv2.countNonZero(frame_diff)
            motion_levels.append(motion_pixels)
            
            # Análisis de brillo
            brightness = np.mean(gray)
            brightness_levels.append(brightness)
            
            prev_gray = gray
        
        cap.release()
        
        # Calcular métricas
        avg_motion = np.mean(motion_levels)
        motion_variance = np.var(motion_levels)
        avg_brightness = np.mean(brightness_levels)
        
        # Determinar tipo de contenido
        content_type = "estatico"
        if avg_motion > 5000:
            content_type = "alto_movimiento"
        elif avg_motion > 1000:
            content_type = "movimiento_moderado"
        
        analysis = {
            'filename': Path(video_path).name,
            'duration_minutes': duration / 60,
            'total_frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'avg_motion': avg_motion,
            'motion_variance': motion_variance,
            'avg_brightness': avg_brightness,
            'content_type': content_type,
            'recommended_strategy': self._recommend_strategy(avg_motion, motion_variance, duration)
        }
        
        return analysis
    
    def _recommend_strategy(self, avg_motion, motion_variance, duration):
        """Recomienda la mejor estrategia basada en el análisis"""
        if duration > 1800:  # Videos muy largos (>30 min)
            return "time_intervals"
        elif avg_motion > 3000 and motion_variance > 1000000:
            return "motion_based"
        elif avg_motion < 500:  # Contenido muy estático
            return "keyframes"
        else:
            return "adaptive"
    
    def extract_time_intervals(self, video_path, output_dir, interval_seconds=5, max_frames=1000):
        """
        Extrae frames en intervalos de tiempo fijos
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_interval = int(fps * interval_seconds)
        
        saved_count = 0
        log_data = []
        
        print(f"⏱️  EXTRACCIÓN POR INTERVALOS: 1 frame cada {interval_seconds}s")
        
        for frame_num in range(0, total_frames, frames_per_interval):
            if saved_count >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                timestamp = frame_num / fps
                filename = f"interval_{saved_count:06d}.jpg"
                output_path = output_dir / filename
                
                cv2.imwrite(str(output_path), frame)
                
                log_data.append({
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'filename': filename,
                    'strategy': 'time_intervals'
                })
                saved_count += 1
        
        cap.release()
        self._save_extraction_log(video_path, log_data, 'time_intervals')
        print(f"✅ {saved_count} frames extraídos (intervalos de {interval_seconds}s)")
        return saved_count
    
    def extract_motion_based(self, video_path, output_dir, motion_threshold=1000, max_frames=800):
        """
        Extrae frames basado en detección de movimiento
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        
        ret, prev_frame = cap.read()
        if not ret:
            return 0
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        saved_count = 0
        log_data = []
        
        print(f"🏃 EXTRACCIÓN POR MOVIMIENTO (umbral: {motion_threshold})")
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret or saved_count >= max_frames:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            motion_pixels = cv2.countNonZero(thresh)
            
            if motion_pixels > motion_threshold:
                timestamp = frame_num / cap.get(cv2.CAP_PROP_FPS)
                filename = f"motion_{saved_count:06d}.jpg"
                output_path = output_dir / filename
                
                cv2.imwrite(str(output_path), frame)
                
                log_data.append({
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'filename': filename,
                    'motion_pixels': motion_pixels,
                    'strategy': 'motion_based'
                })
                saved_count += 1
            
            prev_gray = gray
            frame_num += 1
        
        cap.release()
        self._save_extraction_log(video_path, log_data, 'motion_based')
        print(f"✅ {saved_count} frames con movimiento significativo")
        return saved_count
    
    def extract_adaptive(self, video_path, output_dir, max_frames=600):
        """
        Estrategia adaptativa que combina múltiples técnicas
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Primero analizar el video
        analysis = self.analyze_video_content(video_path)
        if not analysis:
            return 0
        
        print(f"🎯 ESTRATEGIA ADAPTATIVA: {analysis['recommended_strategy']}")
        
        # Distribuir frames entre estrategias
        time_frames = max_frames // 2
        motion_frames = max_frames // 4
        keyframe_frames = max_frames // 4
        
        total_frames = 0
        
        # Extracción por intervalos de tiempo
        time_frames_extracted = self.extract_time_intervals(
            video_path, output_dir / "time_based", 
            interval_seconds=10, max_frames=time_frames
        )
        
        # Extracción por movimiento
        motion_frames_extracted = self.extract_motion_based(
            video_path, output_dir / "motion_based",
            motion_threshold=800, max_frames=motion_frames
        )
        
        # Extracción de frames clave
        keyframe_frames_extracted = self.extract_keyframes(
            video_path, output_dir / "keyframes",
            max_frames=keyframe_frames
        )
        
        total_frames = time_frames_extracted + motion_frames_extracted + keyframe_frames_extracted
        
        print(f"📊 EXTRACCIÓN ADAPTATIVA COMPLETADA:")
        print(f"   ⏱️  Intervalos: {time_frames_extracted} frames")
        print(f"   🏃 Movimiento: {motion_frames_extracted} frames") 
        print(f"   🔑 Frames clave: {keyframe_frames_extracted} frames")
        print(f"   📈 Total: {total_frames} frames")
        
        return total_frames
    
    def extract_keyframes(self, video_path, output_dir, max_frames=300):
        """
        Extrae frames clave usando detección de características
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        
        orb = cv2.ORB_create()
        saved_count = 0
        prev_keypoints = None
        prev_descriptors = None
        log_data = []
        
        print(f"🔑 EXTRACCIÓN DE FRAMES CLAVE")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        for frame_num in range(0, total_frames, frame_interval):
            if saved_count >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Detectar keypoints y descriptores
                keypoints, descriptors = orb.detectAndCompute(frame, None)
                
                # Decidir si es frame clave
                is_keyframe = False
                
                if prev_descriptors is None:
                    is_keyframe = True
                elif descriptors is not None and prev_descriptors is not None:
                    # Comparar descriptores
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(descriptors, prev_descriptors)
                    
                    if len(matches) < len(descriptors) * 0.6:  # Menos del 60% de matches
                        is_keyframe = True
                
                if is_keyframe:
                    timestamp = frame_num / cap.get(cv2.CAP_PROP_FPS)
                    filename = f"keyframe_{saved_count:06d}.jpg"
                    output_path = output_dir / filename
                    
                    cv2.imwrite(str(output_path), frame)
                    
                    log_data.append({
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'filename': filename,
                        'keypoints_count': len(keypoints) if keypoints else 0,
                        'strategy': 'keyframes'
                    })
                    saved_count += 1
                    prev_keypoints = keypoints
                    prev_descriptors = descriptors
        
        cap.release()
        self._save_extraction_log(video_path, log_data, 'keyframes')
        print(f"✅ {saved_count} frames clave extraídos")
        return saved_count
    
    def batch_analyze_videos(self):
        """Analiza todos los videos y genera reporte"""
        videos_dir = self.project_root / "data" / "raw" / "videos"
        videos = list(videos_dir.glob("*.mp4"))
        
        if not videos:
            print("❌ No hay videos para analizar")
            return
        
        analyses = []
        print(f"🔍 ANALIZANDO {len(videos)} VIDEOS...")
        
        for video in videos:
            analysis = self.analyze_video_content(video)
            if analysis:
                analyses.append(analysis)
                print(f"   ✅ {video.name}: {analysis['content_type']}")
        
        # Generar reporte
        self._generate_analysis_report(analyses)
        return analyses
    
    
    
    def smart_batch_process(self, target_frames_per_video=500):
        """
        Procesamiento inteligente de todos los videos
        """
        videos_dir = self.project_root / "data" / "raw" / "videos"
        output_dir = self.project_root / "data" / "processed" / "images" / "train"
        
        videos = list(videos_dir.glob("*.mp4"))
        if not videos:
            print("❌ No hay videos para procesar")
            return
        
        print(f"🧠 PROCESAMIENTO INTELIGENTE DE {len(videos)} VIDEOS")
        print("=" * 60)
        
        total_frames = 0
        processing_log = []
        
        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] PROCESANDO: {video.name}")
            
            # Analizar video
            analysis = self.analyze_video_content(video)
            if not analysis:
                continue
            
            print(f"   📊 Tipo: {analysis['content_type']}")
            print(f"   🎯 Estrategia recomendada: {analysis['recommended_strategy']}")
            
            # Procesar según estrategia recomendada
            if analysis['recommended_strategy'] == 'time_intervals':
                frames = self.extract_time_intervals(
                    video_path=str(video),
                    output_dir=output_dir / self.work_type,
                    interval_seconds=8,
                    max_frames=target_frames_per_video
                )
            elif analysis['recommended_strategy'] == 'motion_based':
                frames = self.extract_motion_based(
                    video_path=str(video),
                    output_dir=output_dir / self.work_type, 
                    motion_threshold=1200,
                    max_frames=target_frames_per_video
                )
            elif analysis['recommended_strategy'] == 'keyframes':
                frames = self.extract_keyframes(
                    video_path=str(video),
                    output_dir=output_dir / self.work_type, 
                    max_frames=target_frames_per_video
                )
            else:  # adaptive
                frames = self.extract_adaptive(
                    video_path=str(video),
                    output_dir=output_dir / self.work_type, 
                    max_frames=target_frames_per_video
                )
            
            processing_log.append({
                'video': video.name,
                'strategy': analysis['recommended_strategy'],
                'frames_extracted': frames,
                'analysis': analysis
            })
            
            total_frames += frames
        
        # Guardar log de procesamiento
        self._save_processing_report(processing_log, total_frames)
        
        print(f"\n🎉 PROCESAMIENTO INTELIGENTE COMPLETADO!")
        print(f"📈 Total frames extraídos: {total_frames}")
        print(f"📁 Directorio: {output_dir / self.work_type}")
        print("\n" + "="*70)
        print("🔄 TRANSICIÓN A ETIQUETADO")
        print("="*70)
        
        proceed_to_labeling = input("\n¿Deseas proceder al etiquetado inteligente de imágenes? (s/n): ").strip().lower()
        
        if proceed_to_labeling == 's':
            print("🚀 Iniciando sistema inteligente de etiquetado...")
            try:
                from src.utils.intelligent_labeling import main as labeling_main
                labeling_main()
            except ImportError as e:
                print(f"❌ Error: No se pudo cargar el sistema de etiquetado: {e}")
                print("💡 Asegúrate de que intelligent_labeling.py esté en src/utils/")
        else:
            print("⚠️  ADVERTENCIA: Sin etiquetado, no podrás entrenar el modelo.")
            print("💡 Puedes ejecutar el etiquetado después con la Opción 5.")
            print(f"📁 Tus frames están en: {output_dir / self.work_type}")
        
        return total_frames
    
    def _save_extraction_log(self, video_path, log_data, strategy):
        """Guarda log de extracción"""
        log_dir = self.project_root / "results" / "processing_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{Path(video_path).stem}_{strategy}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'video': str(video_path),
                'strategy': strategy,
                'extraction_time': datetime.now().isoformat(),
                'frames': log_data
            }, f, indent=2, ensure_ascii=False)
    
    def _save_processing_report(self, processing_log, total_frames):
        """Guarda reporte de procesamiento"""
        report_dir = self.project_root / "data" / "analysis" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_frames_extracted': total_frames,
            'videos_processed': len(processing_log),
            'processing_log': processing_log,
            'summary': {
                'by_strategy': {},
                'total_duration_minutes': sum(log['analysis']['duration_minutes'] for log in processing_log)
            }
        }
        
        # Calcular resumen por estrategia
        for log in processing_log:
            strategy = log['strategy']
            if strategy not in report['summary']['by_strategy']:
                report['summary']['by_strategy'][strategy] = 0
            report['summary']['by_strategy'][strategy] += log['frames_extracted']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Reporte guardado: {report_file}")
    
    def _generate_analysis_report(self, analyses):
        """Genera reporte de análisis de videos"""
        if not analyses:
            return
        
        report_dir = self.project_root / "data" / "analysis" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_videos_analyzed': len(analyses),
            'videos': analyses,
            'summary': {
                'content_types': {},
                'average_duration_minutes': np.mean([a['duration_minutes'] for a in analyses]),
                'total_duration_hours': sum([a['duration_minutes'] for a in analyses]) / 60
            }
        }
        
        # Resumen por tipo de contenido
        for analysis in analyses:
            content_type = analysis['content_type']
            if content_type not in report['summary']['content_types']:
                report['summary']['content_types'][content_type] = 0
            report['summary']['content_types'][content_type] += 1
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Reporte de análisis guardado: {report_file}")

def main():
    processor = AdvancedVideoProcessor()
    
    print("🧠 SISTEMA AVANZADO DE PROCESAMIENTO DE VIDEO")
    print("=" * 50)
    print("1. 🔍 Analizar todos los videos")
    print("2. 🧠 Procesamiento inteligente automático")
    print("3. ⏱️  Extracción por intervalos de tiempo")
    print("4. 🏃 Extracción por movimiento")
    print("5. 🔑 Extracción de frames clave")
    print("6. 🎯 Extracción adaptativa")
    
    choice = input("\nSelecciona opción (1-6): ").strip()
    
    if choice == "1":
        processor.batch_analyze_videos()
    
    elif choice == "2":
        target_frames = int(input("Frames objetivo por video (ej: 500): ") or "500")
        processor.smart_batch_process(target_frames_per_video=target_frames)
    
    elif choice == "3":
        interval = int(input("Segundos entre frames (ej: 5): ") or "5")
        max_frames = int(input("Máximo frames por video (ej: 400): ") or "400")
        processor.batch_process_with_strategy('time_intervals', interval, max_frames)
    
    elif choice == "4":
        processor.batch_process_with_strategy('motion_based')
    
    elif choice == "5":
        processor.batch_process_with_strategy('keyframes')
    
    elif choice == "6":
        processor.batch_process_with_strategy('adaptive')
    
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()