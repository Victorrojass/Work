import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('reto3/photo3.jpg', cv2.IMREAD_GRAYSCALE)

# Dividir la imagen en regiones más grandes (por ejemplo, 50x50 píxeles)
region_size = 5
height, width = image.shape[:2]

# Inicializar variables para el rectángulo que abarcará todas las regiones
min_x = width
min_y = height
max_x = 0
max_y = 0

# Iterar sobre las regiones y encontrar los límites del rectángulo que abarcará todas las regiones
for y in range(0, height, region_size):
    for x in range(0, width, region_size):
        # Obtener la región actual
        region = image[y:y+region_size, x:x+region_size]
        
        # Calcular el promedio de intensidad de la región
        avg_intensity = cv2.mean(region)[0]
        
        # Si el promedio de intensidad es más oscuro que un umbral, considerarlo como parte del agujero
        if avg_intensity < 70:
            # Actualizar los límites del rectángulo
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + region_size)
            max_y = max(max_y, y + region_size)

# Convertir la imagen a color (escala de grises a RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Dibujar un rectángulo alrededor de todas las regiones detectadas (en rojo)
cv2.rectangle(image_rgb, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)  # El color rojo se especifica como (255, 0, 0)

# Mostrar la imagen con el rectángulo que abarca todas las regiones destacadas

image_rgb = cv2.resize(image_rgb, (600, 600))
image = cv2.resize(image, (600, 600))

cv2.imshow("Image original", image)
cv2.imshow('Image with Combined Region', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


