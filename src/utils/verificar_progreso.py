"""
Verificar progreso del proyecto
"""
from pathlib import Path

def verificar_estado():
    print("📊 ESTADO ACTUAL DEL PROYECTO")
    print("=" * 50)
    
    # Verificar videos
    videos_dir = Path("data/raw/videos")
    videos = list(videos_dir.glob("*.mp4"))
    print(f"🎬 VIDEOS: {len(videos)} archivos")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"   📹 {v.name} ({size_mb:.1f} MB)")
    
    # Verificar imágenes
    train_dir = Path("data/processed/images/train")
    images = list(train_dir.glob("*.jpg"))
    print(f"\n📸 IMÁGENES: {len(images)} frames extraídos")
    
    # Verificar anotaciones
    annotations = list(train_dir.glob("*.txt"))
    print(f"🏷️  ANOTACIONES: {len(annotations)} imágenes etiquetadas")
    
    # Calcular progreso
    if images:
        progreso = (len(annotations) / len(images)) * 100
        print(f"\n📈 PROGRESO: {progreso:.1f}% etiquetado")
        
        if progreso < 30:
            print("💡 Necesitas etiquetar más imágenes (mínimo 50)")
        elif progreso < 70:
            print("💡 Buen progreso, continúa etiquetando")
        else:
            print("🎉 ¡Listo para entrenar!")
    
    print(f"\n🎯 SIGUIENTE PASO:")
    if len(annotations) < 50:
        print("   🏷️  Ejecuta 'labelImg' y etiqueta al menos 50 imágenes")
    else:
        print("   🚀 Ejecuta el entrenamiento con 'python run_project.py' → Opción 4")

if __name__ == "__main__":
    verificar_estado()