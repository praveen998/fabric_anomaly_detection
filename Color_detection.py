import cv2
import numpy as np
def closest_color(image):
    image=cv2.imread(image)
    width,height=image.shape[0:2]
    x1=(width//2)-(width//8)
    y1=(height//2)-(height//8)
    x2=(width//2)+(width//8)
    y2=(height//2)+(height//8)
    print(x1,y1,x2,y2)
    image=image[y1:y2,x1:x2]
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rgb=np.mean(image,axis=(0,1)).astype(int)
    #image=image[]
   # red, green, blue = rgb
    colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 0, 255),
    "cyan": (0, 255, 255),
    "brown": (139, 69, 19), 
    }
    closest_color_name = min(colors, key=lambda color: sum((colors[color][i] - rgb[i]) ** 2 for i in range(3)))
    print(closest_color_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
closest_color('dataset/anomaly3.jpeg')