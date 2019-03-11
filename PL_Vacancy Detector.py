from PL.SampleGeneration import ParkingLot
from PL.CarDetection import DetectCars


def main():

    Parking_object = ParkingLot()

	#Positive and negative samples text file
	
    # idx=0
    #
    # #Call create_pos_samples to get the positive text file.
    # dirPath="D:/sunny/image" (image is a parking lot image with many cars)
    # Parking_object.create_pos_samples(dirPath,idx)
    #
    # #Call create_neg_samples to get the negative text file
    # Parking_object.create_neg_samples("D:/negative_samples")

    Detect_object=DetectCars()

    #Call car detection to get the accuracy, TP and FP along with the output image.
    Detect_object.car_detection("lbp",
                                 "D:/PycharmProjects/PKVacancyDetector/LBP/trainclassifier_lbp_3500P_4200N_w21_h20_15S/cascade.xml",
                                 "D:/Downloads/PKLot/PKLot/parking2/cloudy/2012-09-16/2012-10-16_18_23_56.jpg")

if __name__ == "__main__":
    main()

