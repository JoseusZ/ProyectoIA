"""
PRODUCTIVITY MONITOR (v0.1)
Combina YOLOv8 (para objetos) y MediaPipe (para humanos)
para un análisis de contexto avanzado.
"""
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# --- 1. Inicialización ---

# Cargar tu modelo YOLOv8 entrenado (¡asegúrate que la ruta sea correcta!)
# Esta es la salida de tu "Opción 6: Entrenar modelo"
try:
    model = YOLO("results/programador_model/weights/best.pt")
except Exception as e:
    print(f"Error cargando modelo YOLO: {e}")
    print("Asegúrate de haber entrenado el modelo (Opción 6) y que la ruta es correcta.")
    exit()

# Cargar MediaPipe (Manos y Postura)
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Configurar la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

print("🚀 Iniciando monitor... (Presiona 'q' para salir)")

# --- 2. Bucle Principal ---

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. Ejecución de Modelos ---
    
    # A) Ejecutar YOLOv8 (Tracking de Objetos)
    # Usamos .track() para que recuerde los objetos entre frames
    yolo_results = model.track(frame, persist=True, verbose=False)
    
    # El frame con las cajas dibujadas por YOLO
    annotated_frame = yolo_results[0].plot()

    # Convertir el frame a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # B) Ejecutar MediaPipe (Manos y Postura)
    hands_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    # --- 4. Dibujar Resultados de MediaPipe ---
    
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

    # --- 5. Lógica de Productividad (¡El Cerebro!) ---
    # Aquí es donde combinas los datos. (Esto es conceptual)
    
    # Ejemplo:
    # 1. Obtener la caja del 'teclado' desde yolo_results
    # 2. Obtener los puntos (landmarks) de las 'manos' desde hands_results
    # 3. Calcular si los puntos de las manos están DENTRO de la caja del teclado
    # 4. Si es SÍ -> Imprimir en el frame: "ESCRIBIENDO"
    # 5. Si es NO -> Imprimir en el frame: "MANOS FUERA"

    # (Por ahora, solo mostramos los resultados combinados)
    cv2.putText(annotated_frame, "ANALISIS COMBINADO (YOLO + MediaPipe)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 6. Mostrar el Resultado ---
    cv2.imshow("Productivity Monitor", annotated_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 7. Limpieza ---
cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()

def main():
    # Este script se ejecuta directamente, pero puedes llamarlo desde aquí
    print("Ejecutando el monitor de productividad...")
    # (El código principal ya está en el bucle de arriba)

if __name__ == "__main__":
    # Esta estructura permite que el script se ejecute solo
    pass