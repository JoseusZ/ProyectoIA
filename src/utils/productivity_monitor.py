"""
PRODUCTIVITY MONITOR (v0.2 - Multi-Cámara)
Combina YOLOv8 (para objetos) y MediaPipe (para humanos)
para un análisis de contexto avanzado en múltiples cámaras.
"""
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import sys

def main():
    # --- 1. PREGUNTAR AL USUARIO ---
    print("🕵️  Iniciando Monitor de Productividad...")
    try:
        num_cameras_str = input("¿Cuántas cámaras quieres analizar a la vez? (ej: 1, 2, 3): ").strip()
        num_cameras = int(num_cameras_str)
        if num_cameras <= 0:
            print("❌ Número no válido. Debe ser 1 o más.")
            return
    except ValueError:
        print("❌ Entrada no válida. Por favor, introduce un número.")
        return

    # --- 2. INICIALIZACIÓN DE MODELOS ---
    print("Cargando modelos (YOLOv8 y MediaPipe)...")
    try:
        model = YOLO("results/programador_model/weights/best.pt")
    except Exception as e:
        print(f"Error cargando modelo YOLO: {e}")
        print("Asegúrate de haber entrenado el modelo (Opción 6) y que la ruta es correcta.")
        sys.exit(1) # Detener el script si no se puede cargar el modelo

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    # --- 3. INICIALIZACIÓN DE CÁMARAS ---
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
        hands.close()
        pose.close()
        return
        
    num_active_cameras = len(caps)
    print(f"🚀 Iniciando monitor para {num_active_cameras} cámara(s)... (Presiona 'q' para salir)")

    # --- 4. BUCLE PRINCIPAL ---
    while True:
        # Iterar por cada cámara activa
        for i in range(len(caps)):
            cap = caps[i]
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error leyendo frame de cámara {i}. Omitiendo.")
                continue # Saltar al siguiente frame de esta cámara

            # --- 5. Ejecución de Modelos ---
            
            # A) Ejecutar YOLOv8 (Tracking de Objetos)
            yolo_results = model.track(frame, persist=True, verbose=False)
            annotated_frame = yolo_results[0].plot()

            # Convertir el frame a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # B) Ejecutar MediaPipe (Manos y Postura)
            hands_results = hands.process(frame_rgb)
            pose_results = pose.process(frame_rgb)

            # --- 6. Dibujar Resultados de MediaPipe ---
            
            # Dibujar las manos
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS)
                    
            # Dibujar la postura
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS)

            # --- 7. Lógica y Texto ---
            cv2.putText(annotated_frame, f"CAMARA {i}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- 8. Mostrar el Resultado ---
            cv2.imshow(f"Monitor - Cam {i}", annotated_frame)

        # Revisar si se presiona 'q' DESPUÉS de procesar todos los frames de este ciclo
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 9. Limpieza ---
    print("\nCerrando cámaras y limpiando...")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == "__main__":
    # Esta es la forma correcta de hacer que run_project.py lo llame
    main()