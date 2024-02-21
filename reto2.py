import cv2 as cv
import numpy as np

# Cargar la imagen de la caja
img = cv.imread('reto2/photo2.jpg')
object = cv.imread('reto2/object3.png')


cv.imshow('imagen', img)
cv.imshow('object', object)

# Convertir a escala de grises
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_object = cv.cvtColor(object, cv.COLOR_BGR2GRAY)

# Inicializar el detector y el extractor de características AKAZE
akaze = cv.AKAZE_create()

# Encontrar keypoints y descriptores en la imagen de la pegatina
keypoints_sticker, descriptors_sticker = akaze.detectAndCompute(object, None)

# Encontrar keypoints y descriptores en la imagen de la caja
keypoints_box, descriptors_box = akaze.detectAndCompute(gray_img, None)

# Inicializar el matcher de fuerza bruta
bf = cv.BFMatcher()

# Encontrar las mejores coincidencias entre los descriptores de la pegatina y los de la caja
matches = bf.knnMatch(descriptors_sticker, descriptors_box, k=2)

# Aplicar el test de razón de Lowe para filtrar las coincidencias
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extraer las ubicaciones de los keypoints correspondientes a las mejores coincidencias
sticker_pts = np.float32([keypoints_sticker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
box_pts = np.float32([keypoints_box[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calcular la homografía
H, mask = cv.findHomography(sticker_pts, box_pts, cv.RANSAC, 5.0)

# Obtener las esquinas de la imagen de la pegatina
h, w = gray_object.shape
sticker_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# Transformar las esquinas de la imagen de la pegatina utilizando la homografía encontrada
transformed_corners = cv.perspectiveTransform(sticker_corners, H)

# Dibujar el contorno alrededor de la pegatina en la imagen de la caja
img_with_sticker = cv.polylines(img.copy(), [np.int32(transformed_corners)], True, (0, 0, 255), 5)

# Mostrar la imagen con la pegatina detectada
img_with_sticker = cv.resize(img_with_sticker, (500, 500))

cv.imshow('Sticker Detected', img_with_sticker)

cv.waitKey(0)
cv.destroyAllWindows()
