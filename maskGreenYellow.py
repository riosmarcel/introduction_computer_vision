# Importar os pacotes necessarios
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# Construir o analisador de argumento e analisar os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# Definir os limites inferiores e superiores de cada cor

#verde
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#amarelo
yellowLower = (25, 50, 50)
yellowUpper = (32, 255, 255)

# Se um caminho de video nao foi fornecido, pegue a referencia webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# Caso contrario, pegue uma referencia para o arquivo de video
else:
	camera = cv2.VideoCapture(args["video"])

# Manter looping
while True:
	# Agarrar o quadro atual
	(grabbed, frame) = camera.read()

	# Se estamos a ver um video e nos nao pegar um quadro,
	# Em seguida, chegamos ao final do video
	if args.get("video") and not grabbed:
		break

	# Redimensionar o quadro, esbater-lo e converte-lo para o HSV
	# espaco colorido
	frame = imutils.resize(frame, width=600)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Construir uma mascara para a cor verde e outra pra a cor amarela
	# Uma serie de dilatacoes e erosoes para remover qualquer ruido
	maskGreen = cv2.inRange(hsv, greenLower, greenUpper)
	maskGreen = cv2.erode(maskGreen, None, iterations=2)
	maskGreen = cv2.dilate(maskGreen, None, iterations=2)

	maskYellow = cv2.inRange(hsv, yellowLower, yellowUpper)
	maskYellow = cv2.erode(maskYellow, None, iterations=2)
	maskYellow = cv2.dilate(maskYellow, None, iterations=2)

	# Encontrar contornos da mascara e inicializar a corrente
	cntGreen = cv2.findContours(maskGreen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	centerGreen = None
	cntYellow = cv2.findContours(maskYellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	centerYellow = None

	# Unica proceder se pelo menos um contorno foi encontrado
	if len(cntGreen) > 0:
		# Encontrar o maior contorno da mascara, em seguida, usar-lo para calcular o circulo de fecho minima e
		# centroid
		cGreen = max(cntGreen, key=cv2.contourArea)
		rectGreen = cv2.minAreaRect(cGreen)
		boxGreen = cv2.boxPoints(rectGreen)
		boxGreen = np.int0(boxGreen)
		MGreen = cv2.moments(cGreen)
		centerGreen = (int(MGreen["m10"] / MGreen["m00"]), int(MGreen["m01"] / MGreen["m00"]))
		cv2.drawContours(frame, [boxGreen], 0, (0, 255, 0), 2)

	if len(cntYellow) > 0:
		# Encontrar o maior contorno da mascara, em seguida, usar-lo para calcular o circulo de fecho minima e
		# centroid
		cYellow = max(cntYellow, key=cv2.contourArea)
		rectYellow = cv2.minAreaRect(cYellow)
		boxYellow = cv2.boxPoints(rectYellow)
		boxYellow = np.int0(boxYellow)
		MYellow = cv2.moments(cYellow)
		centerYellow = (int(MYellow["m10"] / MYellow["m00"]), int(MYellow["m01"] / MYellow["m00"]))
		cv2.drawContours(frame, [boxYellow], 0, (0, 255, 255), 2)

	# Mostrar o quadro na tela
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Condicao de parada 'q', parar o loop
	if key == ord("q"):
		break

# Limpeza da camara e feche todas as janelas abertas
camera.release()
cv2.destroyAllWindows()
