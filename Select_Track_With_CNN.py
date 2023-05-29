import numpy as np
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import seaborn as sns
import glob
import pickle
import os
import keyboard
import uuid
import cv2
import time
from PIL import Image
from mss import mss
import keyboard
import serial
import struct
from collections import deque
import matplotlib as plt

import warnings

warnings.filterwarnings("ignore")

# %% Servo motor kontrolü için servoyu bağladığımız arduino ile programımız arasında seri iletişim başlatır
seri = serial.Serial("COM3", 9600)
time.sleep(2)

pickle_in = open("model_trained_cnn.p", "rb")
model = pickle.load(pickle_in)

cnnc1 = [0]
cnnc2 = [0]
cnnc3 = [0]
cnnc4 = [0]

# %% Takip algoritmalarımızı 'Sözlük' formatında tanımladık ;
# ## Sözlük
# - bir çeşit karma tablo türüdür
# - anahtar ve değer çiftlerinden oluşur, hemen hemen her variable türü olabilir ama genelde sayılar veya dizilerdir
# - { "anahtar": değer }

# Python programlama dilinde Switch-Case yapısı bulunmaz, algoritmalar arasında seçim yapabilmek için kendi Switch-Case
# mantığımızı oluşturuyoruz, tek bir 'Sözlük' yapısı yeterliydi ancak programı son kullanıcı tarafından daha anlaşılır kılmak
# için 2. bir Sözlük yapısı daha oluşturduk


# Sözlük 1
OPENCV_OBJECT_TRACKERS = {"CSRT": cv2.legacy.TrackerCSRT_create,
                          "KCF": cv2.legacy.TrackerKCF_create,
                          "Boosting": cv2.legacy.TrackerBoosting_create,
                          "Mil": cv2.legacy.TrackerMIL_create,
                          "TLD": cv2.legacy.TrackerTLD_create,
                          "MedianFlow": cv2.legacy.TrackerMedianFlow_create,
                          "Mosse": cv2.legacy.TrackerMOSSE_create}

# Sözlük 2
TRACKERS_KEYS = {"1": "CSRT",
                 "2": "KCF",
                 "3": "Boosting",
                 "4": "Mil",
                 "5": "TLD",
                 "6": "MedianFlow",
                 "7": "Mosse"}

tracker_name = ["Boosting"]
tracker = OPENCV_OBJECT_TRACKERS[tracker_name[0]]()
choice = 0

print('''
      1. CSRT
      2. KCF
      3. Boosting
      4. Mil
      5. TLD
      6. MedianFlow
      7. Mosse
   ''')


def preProcess(imgcnn, masked_cnnc1, masked_cnnc2, masked_cnnc3, masked_cnnc4):
    # blurred = cv2.GaussianBlur(imgcnn,(11,11), 0)
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # black_image = np.zeros(h,w,3),np.uint8
    # black_image[:]=(100,100,0)
    # roi = black_image[y:(y+h),x(x+w)]
    # roi_h,roi_w,_ = roi.shape

    imgcnn = cv2.cvtColor(imgcnn, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(imgcnn.shape[:2], np.uint8)
    mask[masked_cnnc1:int(masked_cnnc1 + masked_cnnc3 / 2), masked_cnnc2:int(masked_cnnc2 + masked_cnnc4 / 2)] = 255
    # mask[0:2500,0:3000] = 255

    imgcnn = cv2.bitwise_and(imgcnn, imgcnn, mask=mask)
    imgcnn = cv2.equalizeHist(imgcnn)
    imgcnn = imgcnn / 255

    return imgcnn


def algoritma(TRACKERS_KEYS, OPENCV_OBJECT_TRACKERS):
    choice = input("Takip algoritmasını sec (1-7)   : ")
    keys = TRACKERS_KEYS.keys()
    if choice in keys:
        tracker_name.append(TRACKERS_KEYS[choice])
        tracker_name.remove(tracker_name[0])
        print(choice, tracker_name)
        # tracker = OPENCV_OBJECT_TRACKERS[tracker_name[len(tracker_name)-1]]()

    else:
        print("! Gecersiz secim ! Varsayilan Takip Algoritmasi {}".format(tracker_name))
    print("Cikis icin 'q'")


algoritma(TRACKERS_KEYS, OPENCV_OBJECT_TRACKERS)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# Cascade
# cap.set(3,frameWidth)
# cap.set(4,frameHeight)
# def empty(a): pass

xlis = [90]
ylis = [90]
zlis = [90]

success, img = cap.read()

# %% Cascade
# cv2.namedWindow("Sonuc")
# cv2.resizeWindow("Sonuc",frameWidth,frameHeight +100)
# cv2.createTrackbar("Scale","Sonuc", 400,1000,empty)
# cv2.createTrackbar("Neighbor","Sonuc",4,50,empty)

cascade = cv2.CascadeClassifier("cascade.xml")


def reel(classIndex, probVal):
    if probVal > 0.85:
        cv2.putText(frame, str(classIndex) + "   " + str(probVal), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        classIndexr = classIndex
        return classIndexr


# %% Obje sınırlarını belirtmek / objeyi kutucuk içerisine almak için oluşturduğumuz fonksiyon
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)

    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), -1)
    cv2.putText(img, "Object At :", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "X =" + str(center_x), (140, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Y =" + str(center_y), (240, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Algoritma : " + tracker_name[0], (400, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.line(img, (640, 360), (center_x, center_y), (255, 0, 255), 1)


def corner_to_mask(bbox, cnnc1, cnnc2, cnnc3, cnnc4):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    s = "x:{}, y:{}, width{}, height:{}".format(np.round(x), np.round(y), np.round(w), np.round(h))
    print(s)

    cnnc1[len(cnnc1) - 1] = x
    cnnc2[len(cnnc2) - 1] = y
    cnnc3[len(cnnc3) - 1] = w
    cnnc4[len(cnnc4) - 1] = h

    return cnnc1, cnnc2, cnnc3, cnnc4


# %% Servo motor kontrolü için oluşturduğumuz fonksiyon
# Servo motorlarımıza gönderilecek olan komutlar
def servo(bbox, classIndexr):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    print(bbox)
    zlis = [90]

    if center_x < 620 and center_x > 660:
        print("kilitlendi_x")
        xlis.append(xlis[len(xlis) - 1])
        xlis.remove(xlis[0])


    elif center_x > 660:
        print("sag")
        xlis.append(xlis[len(xlis) - 1] - 1)
        xlis.remove(xlis[0])

    elif center_x < 620:
        print("sol")
        xlis.append(xlis[len(xlis) - 1] + 1)
        xlis.remove(xlis[0])

    xdeg = xlis[len(xlis) - 1]

    if center_y < 340 and center_y > 380:
        print("kilitlendi_y")
        ylis.append(ylis[len(ylis) - 1])
        ylis.remove(ylis[0])

    elif center_y > 380:
        print("yukari")
        ylis.append(ylis[len(ylis) - 1] + 1)
        ylis.remove(ylis[0])

    elif center_y < 380:
        print("asagi")
        ylis.append(ylis[len(ylis) - 1] - 1)
        ylis.remove(ylis[0])

    ydeg = ylis[len(ylis) - 1]
    s = "x:{}, y:{}, width{}, height:{}".format(np.round(x), np.round(y), np.round(w), np.round(h))
    print(s)

    if classIndexr == 2:

        zlis.append(zlis[len(zlis) - 1] - 1)
        zlis.remove(zlis[0])

    elif classIndexr == 1:

        zlis.append(zlis[len(zlis) - 1] + 1)
        zlis.remove(zlis[0])

    elif classIndexr == 0:
        zlis.append(90)
        zlis.remove(zlis[0])

    zdeg = zlis[len(zlis) - 1]

    # Seri iletişimde motora gönderilen PWM sinyali 0-255 arasında olduğundan 0 dan az 255 den fazla değeri dönderemeyiz
    # Servo motorların dönüş açı sınırları 0-180 derece olduğundan 0 ve 180 derece sınırlarını da aşamazlar
    # Bu nedenle oluşturduğumuz listelerdeki son konumlar 0 dan düşük 180 den fazla olmamalı :
    if xdeg >= 180:
        xlis.append(xlis[len(xlis) - 1] - 1)
        xlis.remove(xlis[0])
        print("x ekseni sınır acısı 180 derece")
    elif xdeg <= 0:
        xlis.append(xlis[len(xlis) - 1] + 1)
        xlis.remove(xlis[0])
        print("x ekseni sınır acısı 0 derece")
    if ydeg >= 180:
        ylis.append(ylis[len(ylis) - 1] - 1)
        ylis.remove(ylis[0])
        print("y ekseni sınır acısı 180 derece")
    elif ydeg <= 0:
        ylis.append(ylis[len(ylis) - 1] + 1)
        ylis.remove(ylis[0])
        print("y ekseni sınır acısı 0 derece")
    if zdeg >= 92:
        zlis.append(zlis[len(zlis) - 1] - 1)
        zlis.remove(zlis[0])
        print("z ekseni sınırı")
    elif zdeg <= 89:
        zlis.append(zlis[len(zlis) - 1] + 1)
        zlis.remove(zlis[0])
        print("z ekseni sınırı ")

    cv2.putText(img, "derece X =" + str(xdeg), (50, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "derece Y =" + str(ydeg), (250, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "derece Z =" + str(zdeg), (460, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Arduino ile çalışmak için aç/kapa #ri.write(struct.pack('>BB', xdeg,ydeg))
    seri.write(struct.pack('>BBB', xdeg, ydeg, zdeg))
    # Kameradan görüntü alma "success" değerini dönderdiği sürece ;


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    success, frame = cap.read()
    success, bbox = tracker.update(img)

    corner_to_mask(bbox, cnnc1, cnnc2, cnnc3, cnnc4)
    s = "x:{}, y:{}, width{}, height:{}".format(np.round(cnnc1), np.round(cnnc2), np.round(cnnc3), np.round(cnnc4))
    masked_cnnc1 = cnnc1[len(cnnc1) - 1]
    masked_cnnc2 = cnnc2[len(cnnc2) - 1]
    masked_cnnc3 = cnnc3[len(cnnc3) - 1]
    masked_cnnc4 = cnnc4[len(cnnc4) - 1]
    imgcnn = np.asarray(frame)
    imgcnn = cv2.resize(imgcnn, (32, 32))
    imgcnn = preProcess(imgcnn, masked_cnnc1, masked_cnnc2, masked_cnnc3, masked_cnnc4)

    imgcnn = imgcnn.reshape(1, 32, 32, 1)

    center = cv2.circle(img, (640, 360), 2, (0, 0, 255), -1)

    if success:

        drawBox(img, bbox)

        classIndex = int(model.predict_classes(imgcnn))
        predictions = model.predict(imgcnn)
        probVal = np.amax(predictions)

        print(classIndex, probVal)
        classIndexr = reel(classIndex, probVal)

        servo(bbox, classIndexr)

    # Cascade
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # #detection parameters
    # scaleVal = 1 + (cv2.getTrackbarPos("Scale","Sonuc")/1000)
    # neighbor = (cv2.getTrackbarPos("Neighbor","Sonuc"))
    # #detection
    # rects = cascade.detectMultiScale(gray,scaleVal,neighbor)

    # for(x,y,w,h) in rects:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
    #     cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)

    else:
        cv2.putText(img, "Nesne Secmek icin : 't'", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Cikis Yapmak icin : 'q'", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Tracking", img)
    cv2.imshow("CNN", frame)
    # Cascade
    # cv2.imshow("Cascade",img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("t"):
        bbox = cv2.selectROI("Tracking", img, False)
        # Seçilen kısım img ve bbox değerini döndürecek
        tracker.init(img, bbox)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()