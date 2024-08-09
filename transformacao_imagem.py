import numpy as np
from PIL import Image

def transformar_imagem(caminho_imagem, matriz_transformacao):
    # Abra a imagem usando Pillow
    imagem = Image.open(caminho_imagem).convert('RGB')

    # Converta a imagem para um array numpy
    array_imagem = np.array(imagem)
    
    # Pega as dimensões da imagem
    altura, largura = array_imagem.shape[:2]

    # Cria uma matriz de coordenadas para os cantos da imagem
    cantos = np.array([
        [0, 0],
        [largura, 0],
        [largura, altura],
        [0, altura]
    ]).T

    # Aplica a transformação aos cantos
    cantos_transformados = matriz_transformacao @ cantos

    # Encontra as novas dimensões da imagem transformada
    min_x = cantos_transformados[0].min()
    max_x = cantos_transformados[0].max()
    min_y = cantos_transformados[1].min()
    max_y = cantos_transformados[1].max()

    nova_largura = int(np.ceil(max_x - min_x))
    nova_altura = int(np.ceil(max_y - min_y))

    # Ajuste de offset para reposicionar a imagem transformada
    offset_x = -min_x
    offset_y = -min_y

    # Nova matriz de transformação com o offset ajustado
    matriz_transformacao_ajustada = np.array([
        [matriz_transformacao[0, 0], matriz_transformacao[0, 1], offset_x],
        [matriz_transformacao[1, 0], matriz_transformacao[1, 1], offset_y],
        [0, 0, 1]
    ])

    # Cria uma matriz de coordenadas para os pixels da nova imagem
    indices_y, indices_x = np.indices((nova_altura, nova_largura))
    coords_homogeneas = np.stack((indices_x.ravel(), indices_y.ravel(), np.ones_like(indices_x).ravel()))

    # Aplica a transformação linear inversa para mapear os pixels da nova imagem para a original
    matriz_inversa = np.linalg.inv(matriz_transformacao_ajustada)
    coords_originais = matriz_inversa @ coords_homogeneas
    coords_originais = coords_originais[:2] / coords_originais[2]

    # Encontra as coordenadas válidas dentro da imagem original
    x_original = np.round(coords_originais[0]).astype(int).reshape(nova_altura, nova_largura)
    y_original = np.round(coords_originais[1]).astype(int).reshape(nova_altura, nova_largura)
    mascara = (x_original >= 0) & (x_original < largura) & (y_original >= 0) & (y_original < altura)

    # Cria uma nova imagem transformada
    array_imagem_transformada = np.zeros((nova_altura, nova_largura, 3), dtype=array_imagem.dtype)
    array_imagem_transformada[mascara] = array_imagem[y_original[mascara], x_original[mascara]]

    # Converte o array numpy de volta para uma imagem Pillow
    imagem_transformada = Image.fromarray(array_imagem_transformada)

    return imagem_transformada

# Caminho para a imagem de entrada
caminho_imagem = 'imagem.jpg'

# Rotação
angulo = np.deg2rad(45)
matriz_rotacao = np.array([
    [np.cos(angulo), -np.sin(angulo)],
    [np.sin(angulo), np.cos(angulo)]
])

# Escalação
escala_x, escala_y = 1, 1
matriz_escala = np.array([
    [escala_x, 0],
    [0, escala_y]
])

# Transformações Múltiplas
matriz_combinada = matriz_rotacao @ matriz_escala

# Aplique a transformação e salve a imagem resultante
imagem_transformada = transformar_imagem(caminho_imagem, matriz_combinada)
imagem_transformada.save('imagem_transformada.jpg')
