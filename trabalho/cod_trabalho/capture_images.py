# ===========================================================================
# Nome completo: Guilherme do Amaral
# RAs: 11202130906
# Data: 30/07/2025
# Nome do Programa: reconhecimento_formas.py
# Descrição: Captura imagens de um padrão de xadrez com a câmera embutida
#            para calibração monocular. As imagens são salvas na pasta 'data/calib'.
# Chamada no terminal: python3 capture_images.py
# ===========================================================================

import cv2
import numpy as np
import time
import os

# === CONFIGURAÇÕES ===
camera_id = 1  # Geralmente 0 para câmera embutida
output_dir = "data/calib"  # Pasta onde as imagens serão salvas
num_images = 15  # Quantas imagens capturar
chessboard_size = (9, 6)  # Número de cantos internos do tabuleiro (largura, altura)

# Criar pasta de saída
os.makedirs(output_dir, exist_ok=True)

# Abrir câmera
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Ajustar resolução (opcional, para melhor qualidade)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Capturando {num_images} imagens para calibração.")
print("Posicione o padrão de xadrez (8x6) na frente da câmera.")
print("Pressione ESC para sair a qualquer momento.\n")

count = 0
start_timer = False

while count < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_copy = frame.copy()

    # Detectar o tabuleiro de xadrez
    ret_chess, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Mostrar feedback visual
    if ret_chess:
        cv2.drawChessboardCorners(img_copy, chessboard_size, corners, ret_chess)
        cv2.putText(img_copy, "Padrao detectado!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Contagem regressiva de 1 segundos antes de capturar
    if start_timer:
        timer = 1 - int(time.time() - start_time)
        if timer > 0:
            cv2.putText(img_copy, str(timer), (300, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
        else:
            # Salvar imagem
            filename = f"{output_dir}/img{count+1:02d}.png"
            cv2.imwrite(filename, frame)
            print(f"Imagem {count+1} salva: {filename}")
            count += 1
            start_timer = False  # Resetar para próxima captura

    # Mostrar contagem no canto
    cv2.putText(img_copy, f"Imagens: {count}/{num_images}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Captura para Calibracao", img_copy)

    key = cv2.waitKey(1) & 0xFF

    # Pressione 'c' para capturar uma nova imagem (se o tabuleiro for detectado)
    if key == ord('c') and ret_chess and not start_timer:
        start_timer = True
        start_time = time.time()
        print(f"Preparando para capturar imagem {count+1}...")

    # Pressione ESC para sair
    if key == 27:
        break

# Finalizar
cap.release()
cv2.destroyAllWindows()
print("Captura finalizada. Use as imagens para calibrar a câmera.")