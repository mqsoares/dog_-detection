import cv2
import numpy as np

# Carregar modelo YOLO
modelConf = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'
cocoNames = 'coco.names'

# Carregar classes do coco.names
classes = []
with open(cocoNames, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

class_id = classes.index('dog')

# Configurar YOLO
net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Captura de vídeo
video = cv2.VideoCapture()
ip = "https://192.168.0.10:8080/video"
video.open(ip)

# Confiança mínima
confThresh = 0.5

# Definir linha de passagem e margem
row = None
margin = 20

# Estado de controle
crosser = False

while True:
    check, img = video.read()
    if not check:
        break

    # Mantém a resolução original
    imH, imW, imC = img.shape

    # Definir linha no topo
    if row is None:
        row = imH // 4

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    bbox = []
    classIds = []
    confs = []

    # Para verificar se o objeto cruzou a linha com margem
    crosser_row = False

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThresh:
                w, h = int(det[2] * imW), int(det[3] * imH)
                x, y = int((det[0] * imW) - w / 2), int((det[1] * imH) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    index = cv2.dnn.NMSBoxes(bbox, confs, confThresh, 0.3)

    for i in index:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        if classIds[i] == class_id:
            # Verifica se o topo da bounding box está acima da linha + margem
            if y < (row - margin):
                crosser_row = True

                # Se entrou agora e antes não estava, emite alerta
                if not crosser:
                    print("⚠️ ALERTA: Passou pela linha!")
                    crosser = True

        # Desenhar retângulo ao redor do objeto detectado
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f'{classes[classIds[i]].upper()} {int(confs[i] * 100)}%'
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Se nada cruzou a linha, reseta o estado
    if not crosser_row:
        crosser = False

    # Desenhar linha de passagem na imagem
    cv2.line(img, (0, row), (imW, row), (255, 0, 0), 2)  # Linha azul
    cv2.line(img, (0, row - margin), (imW, row - margin), (0, 255, 255), 2)  # Linha amarela

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
