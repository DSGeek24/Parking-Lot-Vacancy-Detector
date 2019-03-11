# Parking-Lot-Vacancy-Detector
Detect vacancy spots in a parking lot image. 

Developed an application that detects cars in an image using a LBP and HAAR trained classifier (trained on millions of images of cars 
extracted from parking lot images) and reports the number of vacancy spots left in the parking lot based on the current parking lot image.
The dataset used consists of parking lot images in sunny, cloudy and rainy conditions. XML files with ground truth details are also 
provided with each image. 

The task can be divided into 4 steps: 

a.Create training set: The training dataset consists of both positive and negative samples. Negative samples are the images which do not correspond to cars (non-object images). Positive samples correspond to images with detected cars from parking lots. 
Negative samples are enumerated in a special file “bg.txt” which is created manually. 
Using different functions we obtain positive samples text file where each line has format:
image_path num_obj x y w h  
Here image_path corresponds to path of the image, num_obj corresponds to number of objects in that image which is followed by x, y, w, h (x, y are starting coordinates, w and h are width and height of cropped image).  

b.Training cascade classifier

c. Car detection

d. Parking lot analysis

There are multiple steps involved at each stage with different commands to create positive samples text file, train a LBP and HAAR 
classifier and detect the cars and vacancy spots in a parking lot image. 

Shoot me an email(dkonreddy@uh.edu) for detailed instructions, dataset, findings and results. 

