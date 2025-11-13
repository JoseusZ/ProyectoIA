"""
Recolector de datos simplificado para entrenamiento fácil
"""
import cv2
import os
from pathlib import Path
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        
    def record_session(self, session_name=None, duration=60):
        """Graba una sesión de trabajo automáticamente"""
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🎥 Grabando sesión: {session_name}")
        print(f"⏱️ Duración: {duration} segundos")
        print("💡 Presiona 'q' para terminar antes de tiempo")
        
        # Configurar cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se puede acceder a la cámara")
            print("💡 Conecta una cámara web o verifica los permisos")
            return
        
        # Directorio de salida
        output_dir = self.project_root / "data" / "raw" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{session_name}.mp4"
        
        # Configurar video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        frame_size = (640, 480)
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        
        # Grabar
        start_time = datetime.now()
        frames_recorded = 0
        
        print("🔴 Grabando... Presiona 'Q' para detener")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar para consistencia
            frame = cv2.resize(frame, frame_size)
            
            # Escribir frame
            out.write(frame)
            frames_recorded += 1
            
            # Mostrar preview
            cv2.imshow('Grabando - Presiona Q para terminar', frame)
            
            # Verificar tiempo o tecla
            elapsed = (datetime.now() - start_time).seconds
            if elapsed >= duration or cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Liberar recursos
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Sesión grabada: {frames_recorded} frames")
        print(f"📁 Video: {output_path}")
        
        return output_path
    
    def auto_extract_frames(self, video_path=None):
        """Extrae frames automáticamente para entrenamiento"""
        if video_path is None:
            # Usar el video más reciente
            videos_dir = self.project_root / "data" / "raw" / "videos"
            videos = list(videos_dir.glob("*.mp4"))
            if not videos:
                print("❌ No hay videos grabados")
                return
            video_path = max(videos, key=os.path.getctime)
        
        from utils.advanced_video_processor import VideoProcessor
        
        processor = VideoProcessor()
        frames_extracted = processor.extract_frames(
            video_path=str(video_path),
            output_dir="data/processed/images/train",
            frame_interval=15,
            max_frames=200
        )
        
        print(f"✅ {frames_extracted} frames extraídos para entrenamiento")
        return frames_extracted

def main():
    collector = DataCollector()
    
    print("🎯 RECOLECTOR DE DATOS AUTOMÁTICO")
    print("1. Grabar nueva sesión (60 segundos)")
    print("2. Extraer frames de última sesión")
    
    choice = input("Selecciona opción (1/2): ").strip()
    
    if choice == "1":
        collector.record_session(duration=60)
    elif choice == "2":
        collector.auto_extract_frames()
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()