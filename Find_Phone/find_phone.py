import os
import cv2
import train_phone_finder
import sys
CWD_PATH = os.getcwd()

def main():

    IMAGE_NAME = sys.argv[1]
    #print(IMAGE_NAME)
    #IMAGE_NAME = 'image12.jpg'
    # Path to image
    #PATH_TO_IMAGE = os.path.join(CWD_PATH,'test_images',IMAGE_NAME)
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    image = cv2.imread(PATH_TO_IMAGE)

    box = train_phone_finder.get_graph(image)
    ymin,xmin,ymax,xmax = box[0][0]
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    print(round(x,4),round(y,4))
    # Perform the actual detection by running the model with the image as input
    #print(box[0][0])

main()
