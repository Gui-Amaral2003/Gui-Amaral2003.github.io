# ===========================================================================
# Nome completo: Guilherme do Amaral
# RAs: 11202130906
# Data: 30/07/2025
# Nome do Programa: reconhecimento_formas.py
## Descrição: Sistema de Visão Computacional para estimar a distância de um objeto plano
#            usando calibração da câmera e largura conhecida.
# Chamada no terminal: python3 main.py
# ===========================================================================

import cv2
import numpy as np
import os

def estimate_distance(focal_length_px, real_width_cm, width_pixels):
    """Estima a distância em centímetros."""
    if width_pixels <= 0:
        return 0
    return (real_width_cm * focal_length_px) / width_pixels

# === CARREGAR PARÂMETROS DE CALIBRAÇÃO ===
param_file = "data/monocam_params.xml"
if not os.path.exists(param_file):
    print("Erro: Arquivo de calibração não encontrado.")
    print("Execute primeiro o script de calibração e certifique-se de que o arquivo está em 'data/'.")
    exit()

cv_file = cv2.FileStorage(param_file, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_coeffs = cv_file.getNode("dist_coeffs").mat()
focal_length_px = cv_file.getNode("focal_length_px").real()
cv_file.release()

# # === SIMULAÇÃO DE PARÂMETROS DE CALIBRAÇÃO (para testes) ===
# camera_matrix = np.array([
#     [600, 0, 320],   # f_x, f_y, c_x, c_y
#     [0, 600, 240],
#     [0, 0, 1]
# ])
# dist_coeffs = np.array([0.1, -0.05, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3
# focal_length_px = 600  # Distância focal em pixels

print(f"Matriz da câmera carregada:\n{camera_matrix}")
print(f"Coeficientes de distorção: {dist_coeffs.flatten()}")
print(f"Distância focal (px): {focal_length_px:.2f}")

# Dimensão real do objeto (em cm)
REAL_WIDTH_CM = 2.5  # Exemplo: cartão quadrado de 10x10 cm

# Configurar webcam
cap = cv2.VideoCapture(0)  # Use 0 para câmera embutida, 1 para webcam externa
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Corrigir distorção
ret, frame_test = cap.read()
if not ret:
    print("Erro: Não foi possível acessar a webcam.")
    exit()
h, w = frame_test.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32F)

# Título da janela
NOME_EQUIPE = "Grupo - 6"
window_title = f"Medidor de Distância - {NOME_EQUIPE}"

cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Remover distorção
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # Aproximar contorno
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        # Considerar apenas objetos retangulares
        if len(approx) == 4 and 0.8 < w/h < 1.2:
            forma = "quadrado"
            largura_em_pixels = float(w)

            # Estimar distância
            distancia = estimate_distance(focal_length_px, REAL_WIDTH_CM, largura_em_pixels)
            distancia = max(10.0, min(distancia, 100.0))  # Limitar a 10–500 cm

            # Mostrar resultado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{forma}: {int(distancia)}cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar resultado
    cv2.imshow(window_title, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()