from imutils import face_utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imutils
import time
import dlib
import cv2
import os

def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files

#2D Gaussian function
def twoD_Gaussian((x, y), xo, yo, sigma_x, sigma_y):
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()

# Store the shape_predictors path
predictor_path = os.path.abspath("/main_folder/shape_predictor_68_face_landmarks.dat")
output_path='/main_folder/landmark_results'

# Display Width
display_width = 1000
scale= 1
downscaled_width = display_width / 2

# Load DLibs face detector
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

# Load the shape_predictor
predictor = dlib.shape_predictor(predictor_path)

# Reading image sequence
sequence_path='/data/face_image_sequence'
list_imgs = list_files(sequence_path)

for i, im in enumerate(list_imgs):
    infile = os.path.join(sequence_path, im)
    RGB_frame = cv2.imread(infile)
    frame = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    height, width = frame.shape[:2]
    s_height, s_width = height // scale, width // scale
    imgDim = s_height
    img = cv2.resize(frame, (s_width, s_height))

    # Detect the faces
    rects = detector(img, 0)

    # For each detected face
    for rect in rects:
        # Apply the facial landmarks to the region
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        npLandmarks = np.float32(shape)
        shape = np.delete(shape, [60, 64], axis=0)
        shape = shape[17:]
        N = shape.shape[0]
        # Convert dlib coordinates to OpenCV coordinates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        faceOrig = RGB_frame[y:y + h, x:x + w]
        arr = np.zeros((h, w), np.float)
        arr_2 = np.zeros((height, width), np.float)
        yy, xx = np.mgrid[y:h + y, x:w + x]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Point size is proportional to face size
        radius = (dlib.rectangle.height(rect) * scale / 90)

        # For every landmark coordinate
        for (xs, ys) in shape:
            # Draw the points onto the frame
            cv2.circle(frame, (xs, ys), radius, (0, 0, 255), -1)
            Gauss = twoD_Gaussian((xx, yy), xs, ys, 10, 10)
            map = Gauss.reshape(xx.shape[0], yy.shape[1])
            arr = arr + map / N

        minval = arr.min()
        maxval = arr.max()
        arr -= minval
        arr *= (255.0 / (maxval - minval))
        arr_2[y:h + y, x:w + x] = arr
        map_result = Image.fromarray((arr_2).astype(np.uint8))
        out_path = output_path + str(infile)[-13:-4] + '.png'
        map_result.save(out_path)









