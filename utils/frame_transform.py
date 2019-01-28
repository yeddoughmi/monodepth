import cv2
import numpy as np

def contrast(file):
    img = cv2.imread(file, 1)
    cv2.imshow("Original image",img)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(16,16))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert from BGR to LAB color space
    l, a, b = cv2.split(lab) # split on 3 different channels

    l2 = clahe.apply(l) # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b)) # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # convert from LAB to BGR
    cv2.imshow('Increased contrast', img2)
    cv2.imwrite('../test/test_63_2.jpg', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adjust_gamma(image, gamma=1.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def test_gamma(image_path):
    # load the original image
    original = cv2.imread(image_path)

    # loop over various values of gamma
    for gamma in np.arange(0.0, 3.5, 0.5):
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue

        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(original, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        #adjusted[:,:,0] = 0
        #adjusted[:,:,1] = 0
        #adjusted[:,:,2] = 0

        cv2.imshow("Images", np.hstack([original, adjusted]))
        cv2.waitKey(0)

if __name__ == '__main__':
    test_gamma()
