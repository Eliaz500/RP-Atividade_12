import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carrega a imagem
imagem = cv2.imread('imagens/LUA.png')

# Converte a imagem em cinza
imagem_cinza = cv2.imread('imagens/LUA.png', cv2.IMREAD_GRAYSCALE)

filtro = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Aplicar o filtro de Laplaciano
imagem_filtro = cv2.filter2D(imagem_cinza, -1, filtro)

# Mostra a imagem original e a imagem equalizada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(imagem_filtro, cmap='gray')
plt.title('Imagem Com Filtro')

plt.show()
cv2.waitKey(0)