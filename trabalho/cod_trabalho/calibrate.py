# ===========================================================================
# Nome completo: Guilherme do Amaral
# RAs: 11202130906
# Data: 30/07/2025 (Atualizado em 11/08/2025)
# Nome do Programa: calibrate.py (Versão Corrigida)
# Descrição: Realiza a calibração da camera de forma métrica e com feedback.
# Chamada no terminal: python3 calibrate.py
# ===========================================================================

import cv2
import numpy as np
import glob
import os

# ===========================================================================
# PARÂMETROS - AJUSTE ESTES VALORES
# ===========================================================================

# 1. Dimensões do tabuleiro (número de cantos internos, não de quadrados)
chessboard_size = (9, 6) 

# 2. <<< MUDANÇA IMPORTANTE >>>
# Meça o lado de UM quadrado do seu tabuleiro físico com uma régua e coloque o valor aqui.
# Use a mesma unidade do seu script principal (cm, pois você usa REAL_WIDTH_CM).
SQUARE_SIZE_CM = 2.5 # Exemplo para quadrados de 2.5 cm. MEÇA O SEU E AJUSTE!

# 3. Pasta onde você salvou as imagens de calibração
IMAGES_FOLDER = 'data/calib/'

# ===========================================================================

# Critérios para refinar a localização dos cantos
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar pontos do objeto (3D): coordenadas reais dos cantos
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# <<< MUDANÇA CRÍTICA >>>
# Escalando os pontos do objeto para suas medidas reais em centímetros
objp = objp * SQUARE_SIZE_CM

# Arrays para armazenar os pontos do objeto (3D) e os pontos da imagem (2D)
obj_points = []  # Pontos 3D no mundo real
img_points = []  # Pontos 2D na imagem

images = glob.glob(os.path.join(IMAGES_FOLDER, '*.png'))
if not images:
    print(f"Erro: Nenhuma imagem encontrada em '{IMAGES_FOLDER}'. Verifique o caminho e a extensão do arquivo.")
    exit()

print(f"Processando {len(images)} imagens...")

frame_size = None # Será definido a partir da primeira imagem

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # <<< MUDANÇA >>> Definir o tamanho do frame dinamicamente
    if frame_size is None:
        frame_size = gray.shape[::-1]

    # Encontrar os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print(f"  -> Cantos encontrados em: {os.path.basename(fname)}")
        obj_points.append(objp)
        # Refinar os cantos encontrados
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_refined)
    else:
        print(f"  -> AVISO: Cantos não encontrados em {os.path.basename(fname)}. Imagem descartada.")

if not obj_points:
    print("\nERRO FATAL: Não foi possível detectar o tabuleiro em NENHUMA imagem. A calibração não pode continuar.")
    exit()

print("\nCalibrando a câmera... Por favor, aguarde.")

# Calibrar
# <<< MUDANÇA >>> Passando o frame_size dinâmico
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, frame_size, None, None)

# <<< MUDANÇA IMPORTANTE >>> Calcular e exibir o erro de reprojeção
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
reprojection_error = mean_error / len(obj_points)

# Salvar os parâmetros
os.makedirs("data", exist_ok=True)
output_file = "data/monocam_params.xml"
cv_file = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", camera_matrix)
cv_file.write("dist_coeffs", dist_coeffs)
cv_file.write("focal_length_px", camera_matrix[0, 0]) # fx
cv_file.write("reprojection_error", reprojection_error)
cv_file.release()

# Exibir os resultados
print("\n--- CALIBRAÇÃO CONCLUÍDA ---")
print(f"Parâmetros salvos em '{output_file}'")
print(f"Matriz da Câmera (K):\n{camera_matrix}")
print(f"\nCoeficientes de Distorção:\n{dist_coeffs.ravel()}")
print(f"\nErro médio de reprojeção: {reprojection_error:.4f} pixels")
print("--> (Idealmente, este valor deve ser < 0.5. Se for > 1.0, refaça a captura das fotos com mais qualidade e variedade).")