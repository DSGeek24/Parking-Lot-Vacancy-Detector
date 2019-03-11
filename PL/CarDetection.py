import cv2
import numpy as np
import os, os.path
import xml.etree.ElementTree as ET
from datetime import datetime

class DetectCars:

    #Method used to display the image with detected cars and also to find TP,FP and Accuracy
    def car_detection(self,feature,cascadeFile,testimage):
        No_Occupied_ParkingSpots = 0
        TP =0
        detected_cars=[]
        if(feature=="lbp"):
            lbp_cascade = cv2.CascadeClassifier(cascadeFile)
            img = cv2.imread(testimage)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            car = lbp_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in car:
                single_car=[]
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                single_car.append(x)
                single_car.append(y)
                single_car.append(w)
                single_car.append(h)
                detected_cars.append(single_car)
            cv2.imwrite("output_images/output_lbp"+ "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg", img)

        if(feature=="haar"):
            haar_cascade = cv2.CascadeClassifier(cascadeFile)
            img = cv2.imread(testimage)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            car = haar_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in car:
                single_car = []
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                single_car.append(x)
                single_car.append(y)
                single_car.append(w)
                single_car.append(h)
                detected_cars.append(single_car)
            cv2.imwrite("output_images/output_haar"+ "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg", img)

        xmlFile = testimage.replace(".jpg", ".xml")

        tree = ET.parse(xmlFile)
        root = tree.getroot()
        contours = []
        for child in root:
            if (len(child.attrib.keys()) > 1):
                if (child.attrib['occupied'] == '1'):
                    No_Occupied_ParkingSpots=No_Occupied_ParkingSpots+1
                    single_contour = []
                    try:
                        for point in child.iter('point'):
                            points = []
                            x = int(point.attrib['x'])
                            y = int(point.attrib['y'])
                            points.append(x)
                            points.append(y)
                            single_contour.append(points)
                        contours.append(np.array(single_contour))
                    except(RuntimeError):
                        print("error")

        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            ground_truth_box=[]
            ground_truth_box.append(int(rect[0][0]))
            ground_truth_box.append(int(rect[0][1]))
            ground_truth_box.append(int(rect[1][0]))
            ground_truth_box.append(int(rect[1][1]))
            for j in range(len(detected_cars)):
                if(self.bb_intersection_over_union(ground_truth_box,detected_cars[j])>0.5):
                    TP=TP+1
                    break

        FP = len(detected_cars)-TP
        print("Number of true positives are {}".format(TP))
        print("Number of false positives are {}".format(FP))
        accuracy = TP/ No_Occupied_ParkingSpots
        print("Accuracy for feature "+feature+ "is {}".format(accuracy))

    def bb_intersection_over_union(self,ground_truth_box,pred_box):
        xA = max(ground_truth_box[0], pred_box[0])
        yA = max(ground_truth_box[1], pred_box[1])
        xB = min(ground_truth_box[2], pred_box[2])
        yB = min(ground_truth_box[3], pred_box[3])
        interArea = (xB - xA + 1) * (yB - yA + 1)
        ground_truth_box_Area = (ground_truth_box[2] - ground_truth_box[0] + 1) * (ground_truth_box[3] - ground_truth_box[1] + 1)
        pred_box_Area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
        iou = interArea / float(ground_truth_box_Area + pred_box_Area - interArea)
        return iou