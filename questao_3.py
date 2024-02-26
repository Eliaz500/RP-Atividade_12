import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('imagens/Fig0340(a)(dipxe_text).png', cv2.IMREAD_GRAYSCALE)

# convertida para float
imagem_float32 = np.float32(imagem)

# Aplicar o operador de Roberts na direção horizontal
operador_x = cv2.filter2D(imagem_float32, -1, np.array([[1, 0], [0, -1]]), borderType=cv2.BORDER_REPLICATE)
operador_y = cv2.filter2D(imagem_float32, -1, np.array([[0, 1], [-1, 0]]), borderType=cv2.BORDER_REPLICATE)

# Calcular a magnitude dos gradientes direcionais de Roberts
magnitude_gradiente = cv2.magnitude(operador_x, operador_y)

# Converter a magnitude para o tipo de dados uint8
imagem_magnitude_gradiente = np.uint8(magnitude_gradiente)

# Mostra a imagem original e a imagem equalizada
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(imagem_magnitude_gradiente, cmap='gray')
plt.title('Imagem com Magnitude dos gradientes de Roberts')

plt.suptitle('')
plt.show()
cv2.waitKey(0)