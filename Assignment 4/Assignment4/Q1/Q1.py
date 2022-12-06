import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import tensorflow
import camera_matrix_calculation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()

gen = idg.flow_from_directory("./image",target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout

inlayer = Input(shape=(200,200,3))
pp1 = Conv2D(12, 3, activation="relu")(inlayer)
hp1 = MaxPooling2D(2)(pp1)
pp2 = Conv2D(24, 3, activation="relu")(hp1)
hp2 = MaxPooling2D(2)(pp2)
pp3 = Conv2D(36, 3, activation="relu")(hp2)
hp3 = MaxPooling2D(2)(pp3)
pp4 = Conv2D(48, 3, activation="relu")(hp3)
hp4 = MaxPooling2D(2)(pp4)
flat = Flatten()(inlayer)
a1 = Dense(800, activation="relu")(flat)
a2 = Dense(600, activation="relu")(a1)
a3 = Dense(400, activation="relu")(a2)
a4 = Dense(200, activation="relu")(a3)
a5 = Dense(100, activation="relu")(a4)
out_layer = Dense(2, activation="softmax")(a5)

model = Model(inlayer, out_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(gen, epochs=10)

filename ="prachi.mov"
capture = cv2.VideoCapture(filename)

def getClassName(index):
    if index == 0:
        return "object found"
    else:
        return "object not found"

font = cv2.FONT_HERSHEY_COMPLEX
org = (50, 50)

cam_matx = camera_matrix_calculation.camera_matx

cam_matx_inv = np.linalg.inv(cam_matx)

foundImg = cv2.imread("./image/found/capture_isp_1.png")
res,coords = model.predict(np.array([foundImg]))

(fX, fY, fW, fH) = coords

size = "(" + str(fW) + "," + str(fH) + ")"

dimension_matrix = cam_matx_inv.dot(project_points)
size = "(" + str(-1*dimension_matrix[0][0]) + "," + str(-1*dimension_matrix[1][0]) + ")"

fontScale = 1

color = (255, 0, 0)  

thickness = 2

def img_alignment(image_1, image_2):
    image_1, image_2 = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY) 
    img_size = image_1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(image_2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(image_2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img2_aligned

while True:
    _, image_1 = capture.read()
    _, image_2 = capture.read()

    res = model.predict(np.array([cv2.resize(image_1, (200,200))]))
    
    image_1 = cv2.putText(image_1, getClassName(res.argmax(axis=1)) , org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    if getClassName(res.argmax(axis=1))=="object found":
        image_1 = cv2.putText(image_1,size,(100,100), font,fontScale, color, thickness, cv2.LINE_AA)

    diff = cv2.absdiff(image_1, image_2)
    
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

    _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, b, l = cv2. boundingRect(contour)
        if cv2.contourArea(contour) > 300:
            cv2.rectangle(image_1, (x, y), (x+b, y+l), (0,255,0), 2)
    
    cv2.imshow("Motion", image_1)
    key = cv2.waitKey(1)
    if key%256 == 27:
        print("Closing program")
        exit()
