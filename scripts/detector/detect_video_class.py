from class_detector import *
import cv2 as cv
import os


root = '../../../det_class_yolo/inference/images'
# root = '../../../demoset/images'
files = os.listdir(root)
#'iter.pckl'
dt_main = load_state('main.pckl')
# dt_iter = load_state('iter.pckl')
dt_iter = load_state('all_classes.pckl')

# with open('../demoset/train_4.txt', 'r') as f:
#     files = f.readlines()
f = os.listdir('inference/video')

cap = cv.VideoCapture('inference/video/' + f[0])

while cap.isOpened():

    ret, img = cap.read()

    # print(f)
    # img = cv.imread(root + '/' + f)
    # img = cv.imread(f[:-1])

    out_img_main = dt_main.detect(img)[0]
    out_img_iter = dt_iter.detect(img)[0]
    #
    cv.imshow('main', out_img_main)
    cv.imshow('iter', out_img_iter)

   # dt_main.find_object('Box', img)
   # dt_main.find_object('can', img)

    cv.waitKey(1)




cv.destroyAllWindows()