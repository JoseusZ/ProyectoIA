"""
PRODUCTIVITY MONITOR (v0.3 - Global y Configurable)
Combina YOLOv8 y MediaPipe dinámicamente basado
en el archivo 'work_config.yaml'.
"""
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import sys
import yaml  # <--- ¡NUEVO!
from pathlib import Path # <--- ¡NUEVO!

def load_config():
    """Carga el archivo work_config.yaml"""
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "work_config.yaml"
    
    if not config_path.exists():
        print("❌ Error: 'configs/work_config.yaml' no encontrado.")
        print("💡 Ejecuta la 'Opción 1: Configurar...' primero.")
        return None
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config, project_root

def main():
    # --- 1. Cargar Configuración ---
    config, project_root = load_config()
    if config is None:
        sys.exit(1)
        
    work_type = config.get('work_type', 'default')
    mp_modules_to_load = config.get('mediapipe_modules', ['hands', 'pose']) # Default

    # --- 2. PREGUNTAR AL USUARIO (Multi-Cámara) ---
    print(f"🕵️  Iniciando Monitor para: {work_type.upper()}")
    try:
        num_cameras_str = input("¿Cuántas cámaras quieres analizar a la vez? (ej: 1, 2, 3): ").strip()
        num_cameras = int(num_cameras_str)
        if num_cameras <= 0:
            print("❌ Número no válido. Debe ser 1 o más.")
            return
    except ValueError:
        print("❌ Entrada no válida. Por favor, introduce un número.")
        return

    # --- 3. INICIALIZACIÓN DE MODELOS (Dinámico) ---
    print("Cargando modelos dinámicamente...")
    
    # A) Cargar modelo YOLOv8 basado en el config
    model_path = project_root / f"results/{work_type}_model/weights/best.pt"
    
    if not model_path.exists():
        print(f"❌ Error fatal: No se encuentra el modelo entrenado en:")
        print(f"   {model_path}")
        print(f"💡 Asegúrate de haber ejecutado la 'Opción 6: Entrenar' para '{work_type}'.")
        sys.exit(1)
        
    try:
        model = YOLO(model_path)
        print(f"✅ Modelo YOLO '{work_type}' cargado desde {model_path}")
    except Exception as e:
        print(f"Error cargando modelo YOLO: {e}")
        sys.exit(1)

    # B) Cargar módulos de MediaPipe basados en el config
    mp_drawing = mp.solutions.drawing_utils
    
    # Inicializar trackers en None
    hands = None
    pose = None
    face_mesh = None
    
    if 'hands' in mp_modules_to_load:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        print("✅ Módulo MediaPipe 'Hands' cargado.")
        
    if 'pose' in mp_modules_to_load:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5)
        print("✅ Módulo MediaPipe 'Pose' cargado.")
        
    if 'face_mesh' in mp_modules_to_load:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
        print("✅ Módulo MediaPipe 'Face Mesh' cargado.")

    # --- 4. INICIALIZACIÓN DE CÁMARAS ---
    caps = []
    print(f"Intentando abrir {num_cameras} cámara(s)...")
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            caps.append(cap)
            print(f"✅ Cámara {i} abierta exitosamente.")
        else:
            print(f"❌ Error: No se pudo abrir la cámara {i}. Se omitirá.")
    
    if not caps:
        print("No se pudo abrir ninguna cámara. Saliendo.")
        if hands: hands.close()
        if pose: pose.close()
        if face_mesh: face_mesh.close()
        return
        
    print(f"🚀 Iniciando monitor... (Presiona 'q' para salir)")

    # --- 5. BUCLE PRINCIPAL ---
    while True:
        for i in range(len(caps)):
            cap = caps[i]
            ret, frame = cap.read()
            if not ret:
                print(f"Error leyendo frame de cámara {i}. Omitiendo.")
                continue

            # A) Ejecutar YOLOv8 (Tracking de Objetos)
            yolo_results = model.track(frame, persist=True, verbose=False)
            annotated_frame = yolo_results[0].plot()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # B) Ejecutar MediaPipe (Solo módulos cargados)
            if hands:
                hands_results = hands.process(frame_rgb)
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if pose:
                pose_results = pose.process(frame_rgb)
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if face_mesh:
                face_mesh_results = face_mesh.process(frame_rgb)
                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # --- 6. Lógica y Texto ---
            cv2.putText(annotated_frame, f"CAMARA {i} - {work_type.upper()}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f"Monitor - Cam {i}", annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 7. Limpieza ---
    print("\nCerrando cámaras y limpiando...")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    if hands: hands.close()
    if pose: pose.close()
    if face_mesh: face_mesh.close()

if __name__ == "__main__":
    main()