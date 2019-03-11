import cv2
import numpy as np
import os, os.path
import xml.etree.ElementTree as ET
from datetime import datetime

class ParkingLot:

    #Method used to extract the car images and create a positive text file with the following format
    #image_path number_of_objects x y w h
    def create_pos_samples(self,dirPath,idx):
        imageDir = dirPath
        xml_path_list = []
        valid_image_extensions = [".xml"]
        valid_image_extensions = [item.lower() for item in valid_image_extensions]

        for file in os.listdir(imageDir):
            extension = os.path.splitext(file)[1]
            if extension.lower() not in valid_image_extensions:
                continue
            xml_path_list.append(os.path.join(imageDir, file))

        for xmlPath in xml_path_list:
            with open("testSamples.txt", "a") as textfile:
                print(' ',file=textfile)
                carCount=0

            image=xmlPath.replace(".xml",".jpg")
            im = cv2.imread(image)

            tree=ET.parse(xmlPath)
            root=tree.getroot()
            contours = []
            for child in root:
                if(len(child.attrib.keys())>1):
                    if(child.attrib['occupied']=='1'):
                        single_contour = []
                        try:
                            for point in child.iter('point'):
                                points = []
                                x=int(point.attrib['x'])
                                y=int(point.attrib['y'])
                                points.append(x)
                                points.append(y)
                                single_contour.append(points)
                            contours.append(np.array(single_contour))
                            carCount = carCount + 1
                        except(RuntimeError):
                            print("error")
            print(carCount)
            x_list=[]
            y_list = []
            w_list = []
            h_list = []
            print(len(contours))

            for i in range(len(contours)):
                rect=cv2.minAreaRect(contours[i])
                contours_box = cv2.boxPoints(rect)
                x, y = [], []
                for contour_line in contours_box:
                    x.append(contour_line[0])
                    y.append(contour_line[1])
                x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
                car = im[int(y1):int(y2), int(x1):int(x2)]

                #starting co ordinates
                x_list.append(int(x1))
                y_list.append(int(y1))

                #width and height of cropped image
                w_list.append(car.shape[1])
                h_list.append(car.shape[0])

                output_dir = 'images/'
                output_path = output_dir + "car" + str(idx) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
                #cv2.imwrite(output_path, car)
                idx=idx+1

            with open("testSamples.txt", "a") as textfile:
                textfile.write(image+"{}".format("\t"))
                textfile.write(str(carCount)+"{}".format("\t"))

            with open("testSamples.txt", "a") as textfile:
                for i in range(carCount):
                    print("{}"' '"{}"' '"{}"' '"{}".format(x_list[i], y_list[i], w_list[i], h_list[i]),end='', flush=True, file=textfile)
                    print("{}".format("\t"),end='', flush=True,file=textfile)

    #Method used to create the negative samples text file which contains only the image path
    def create_neg_samples(self,dataset):
        image_list=[]
        for file in os.listdir(dataset):
            extension = os.path.splitext(file)[1]
            if extension.lower() in ["jpg"]:
                continue
            image_list.append(os.path.join(dataset, file))

        for image_path in image_list:
            with open("bg.txt", "a") as textfile:
                textfile.write(image_path)
                print(file=textfile)