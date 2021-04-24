import cv2 as cv
import time
import os
import glob 
import Hand_Tracking_Module as htm
import mediapipe as mp

wCam, hCam= 640, 480 

cap=cv.VideoCapture(0)
#define height and width
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "pictures"
List=os.listdir(folderPath)
print(List)

one=cv.imread('/home/lferda/ris_project/pictures/1.jpeg')
two=cv.imread('/home/lferda/ris_project/pictures/2.png')
three=cv.imread('/home/lferda/ris_project/pictures/3.jpg')
four=cv.imread('/home/lferda/ris_project/pictures/4.jpg')
five=cv.imread('/home/lferda/ris_project/pictures/5.png')

numwidth = 200
numheight = 200

Images = glob.glob('/home/lferda/ris_project/pictures/*.jpg')
overlayList = []


for i in Images:
    img = cv.imread(i)
    img = cv.resize(img, (numwidth, numheight), interpolation = cv.INTER_AREA)
    overlayList.append(img)
    




for imPath in List: 
    image=cv.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')


#set time for fps
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [ 4, 8, 12, 16, 20]

while True: 
    success, img = cap.read()
    img= detector.findHands(img)
    
    lmList =detector.findPosition(img, draw=False)
    #print(lmList)
    
    fingers = []

    if len(lmList) != 0:
        
        
        #thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
                
        
        
        
        #4 fingers
        for id in range(0,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
                
        print(fingers)
        
        if fingers[2]==1:
            print("Batata")

            
    totalfinger = fingers.count(1) - 1
    print(totalfinger)

    
    
    
    h, w, c = overlayList[0].shape
    img[0:h, 0:w] = overlayList[totalfinger -1]
    
    cTime = time.time()
    fps = 1/(cTime -pTime)
    pTime = cTime
    
    cv.putText(img, f'FPS: {int(fps)}', (450,70), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0,0), 3)
    cv.rectangle(img, (20,255), (170, 425), (255,50,66), cv.FILLED)
    cv.putText(img, str(totalfinger), (45, 370), cv.FONT_HERSHEY_COMPLEX_SMALL, 10, (80, 280, 0), 25)
    cv.imshow("Fingers", img)
    cv.waitKey(1)
    
    
    if cv.waitKey(20) & 0xFF==ord('s'):
        break
    
    