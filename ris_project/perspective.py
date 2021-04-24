import cv2 as cv
import numpy as np  

capture = cv.VideoCapture('road_car_view.mp4')

while True: 
    _, frame = capture.read()
    
    cv.circle(frame, (550, 460), 5, (0, 0 , 255), -1)
    cv.circle(frame, (150, 720), 5, (0, 0 , 255), -1)
    cv.circle(frame, (1200,720), 5, (0, 0 , 255), -1)
    cv.circle(frame, (770, 460), 5, (0, 0 , 255), -1)
    
    
    src = np.float32([(550, 460), (150, 720), (1200,720), (770, 460)])
    dst = np.float32([(100, 0), (100,720), (1100, 720), (1100, 0)])
   
   
   
    M=cv.getPerspectiveTransform(src, dst)
    
    result = cv.warpPerspective(frame, M, (1200, 720))
    
    cv.imshow('frame', frame )
    cv.imshow("Perspective", result)
    
    if cv.waitKey(20) & 0xFF==ord('s'):
        break
    
    
    
src = np.float32([(550, 460), (150, 720), (1200,720), (770, 460)])
dst = np.float32([(100, 0), (100,720), (1100, 720), (1100, 0)])
 #M=cv.getPerspectiveTransform(src, dst)
    #warped=cv.warpPerspective(frame, M, (1100, 720))