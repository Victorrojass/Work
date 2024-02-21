import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('reto3/photo3.jpg', cv2.IMREAD_GRAYSCALE)

# Dividir la imagen en regiones
region_size = 5
height, width = image.shape[:2]

# Inicializar variables rectangulo
min_x = width
min_y = height
max_x = 0
max_y = 0

# Iterar sobre las regiones y encontrar cambios de intensidad
for y in range(0, height, region_size):
    for x in range(0, width, region_size):
        # Obtener la región actual
        region = image[y:y+region_size, x:x+region_size]
        
        avg_intensity = cv2.mean(region)[0]
        
        # Ajustar el umbral
        if avg_intensity < 70:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + region_size)
            max_y = max(max_y, y + region_size)

image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Dibujar un rectángulo alrededor de todas las regiones detectadas
cv2.rectangle(image_rgb, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)


image_rgb = cv2.resize(image_rgb, (600, 600))
image = cv2.resize(image, (600, 600))

cv2.imshow("Image original", image)
cv2.imshow('Image with Combined Region', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


