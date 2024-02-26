import cv2
from matplotlib import pyplot as plt

# Carrega a imagem
imagem = cv2.imread('imagens/Fig0340(a)(dipxe_text).png')

# Converte a imagem em cinza
imagem_cinza = cv2.imread('imagens/Fig0340(a)(dipxe_text).png', cv2.IMREAD_GRAYSCALE)

# Aplicar um filtro gaussiano para suavização
imagem_baixa_frequencia = cv2.GaussianBlur(imagem_cinza, (5, 5), 3)

# Definir o fator de realce
k = 9000001

# Calcular a imagem realçada (filtragem de alto reforço)
imagem_filtrada = cv2.addWeighted(imagem_cinza, 1 + k, imagem_baixa_frequencia, -k, 0)

# Mostra a imagem original e a imagem equalizada
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 3, 2)
plt.imshow(imagem_baixa_frequencia, cmap='gray')
plt.title('Imagem Com Filtro')

plt.subplot(1, 3, 3)
plt.imshow(imagem_filtrada, cmap='gray')
plt.title('Imagem Com Filtro Alto Reforço ')

plt.suptitle(f'K = {k}')
plt.show()
cv2.waitKey(0)