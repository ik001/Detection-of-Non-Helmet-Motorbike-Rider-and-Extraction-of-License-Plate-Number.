import cv2
import numpy as np
#................................
i=0
net = cv2.dnn.readNet("yolov3_custom.cfg",r"yolov3_custom_2000.weights")
classes = ['pwb']

net1 = cv2.dnn.readNet("yolov3_custom_6000.weights","yolov3_custom1.cfg")

net2 = cv2.dnn.readNet("yolov3_custom2000np.cfg",r"y3num_plt_dec2000.weights")
#.....................
img = cv2.imread("img1.jpg")#img2
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
ari = []
if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = (255,0,0)
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
        crop_img = img[y:y+h, x:x+w]
        ari.append(crop_img)
        i = i+1
cv2.imshow('Image', img)
#..................
def capnumplt(img):
    img2 = img
    classes2 = ['bnp']
    img2 = cv2.resize(img2,(1280,720))
    hight2,width2,_ = img2.shape
    blob2 = cv2.dnn.blobFromImage(img2, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net2.setInput(blob2)

    output_layers_name2 = net2.getUnconnectedOutLayersNames()

    layerOutputs2 = net2.forward(output_layers_name2)
    boxes2 =[]
    confidences2 = []
    class_ids2 = []

    for output in layerOutputs2:
        for detection in output:
            score2 = detection[5:]
            class_id2 = np.argmax(score2)
            confidence2 = score2[class_id2]
            if confidence2 > 0.5:
                center_x2 = int(detection[0] * width2)
                center_y2 = int(detection[1] * hight2)
                w2 = int(detection[2] * width2)
                h2 = int(detection[3]* hight2)
                x2 = int(center_x2 - w2/2)
                y2 = int(center_y2 - h2/2)
                boxes2.append([x2,y2,w2,h2])
                confidences2.append((float(confidence2)))
                class_ids2.append(class_id2)
    indexes2 = cv2.dnn.NMSBoxes(boxes2,confidences2,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors2 = np.random.uniform(0,255,size =(len(boxes2),3))
    if  len(indexes2)>0:
        for i in indexes2.flatten():
            x2,y2,w2,h2 = boxes2[i]
            label2 = str(classes2[class_ids2[i]])
            confidence2 = str(round(confidences2[i],2))
            color2 = colors2[i]
            cv2.rectangle(img2,(x2,y2),(x2+w2,y2+h2),color2,2)
            cv2.putText(img2,label2 + " " + confidence2, (x2,y2+400),font,2,color2,2)
    cv2.namedWindow("Num_plate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Num_plate", 300, 600)
    cv2.imshow('Num_plate',img2)

#......                
def detecthelmet(ari):
    classes = ['pwb','Helmet','Vehicle registration plate']
    names=['Helmet']
    zz = 0
    for e in ari:
        img = e
        height, width, _ = img.shape

        blob1 = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net1.setInput(blob1)
        output_layers_names1 = net1.getUnconnectedOutLayersNames()
        layerOutputs1 = net1.forward(output_layers_names1)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs1:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        if len(indexes)>0:
            for i in indexes.flatten():
                if str(classes[class_ids[i]]) not in names:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = (255,0,0)
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.imshow('Image{}'.format(zz), img)
                    zz=zz+1
                    capnumplt(img)
                    
detecthelmet(ari)
cv2.waitKey(100000)

