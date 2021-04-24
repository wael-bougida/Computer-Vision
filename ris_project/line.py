import cv2 as cv
import numpy as np
#import rospy 
#from sensor_msgs.msg import CompressedImage
import sys
#from picamera import PiCamera
#from duckietown_msgs.msg import WheelsCmdStamped
#from cv_bridge import CvBridge
#from sensor_msgs.msg import Image






###############d etect white, yellow, red############################# 

#camera  = PiCamera() #### or use Videocapture from cv2 ################# 

camera = cv.VideoCapture('road_car_view.mp4')
#camera = cv.imread(Filaname)




#camera.resolution = (640, 360)
#capture = PiRGBarray(camera, size=(640, 360))
#cv.namedWindow('l3ibat', cv.WINDOW_NORMAL)


###########Add a call to the CvBridge######################### => 
#bridge = CvBridge()
#imgMsg = bridge.cv2_to_imgmsg(img, "bgr8")
def deteyat():
    while True:
        isTrue, orig_frame = camera.read() #capture in case of PI

    frame = cv.GaussianBlur(orig_frame, (5, 5), 0)
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    
    sensitivity = 45
    Whiteline = cv.inRange(hsv, (0, 0, 255-sensitivity), (255, sensitivity, 255))
    white=cv.bitwise_and(frame, frame, mask=Whiteline)
    
    #low_yellow = np.array([18, 94, 140])
    #up_yellow = np.array([48, 255, 255])
    
    yellowline= cv.inRange(hsv, (18, 94, 140), (48, 255, 255))
    yellow=cv.bitwise_and(frame, frame, mask=yellowline)

    #low_red = np.array([161, 155, 84])
    #high_red = np.array([179, 255, 255])
    
    redline = cv.inRange(hsv, (161, 155, 84), (179, 255, 255))
    red = cv.bitwise_and(frame, frame, mask=redline)
    
    lines = np.concatenate((yellow, white, red), axis=0)
    linescalled= cv.resize(lines, (800, 940), interpolation=cv.INTER_AREA)
    
    
    
    cv.imshow('lines', linescalled)
    cv.imshow('frame', frame)
    
    
    #pub = rospy.Publisher('image', Image, queue_size=10)
    #while not rospy.is_shutdown():
    #pub.publish(imgMsg)
    #rospy.Rate(1.0).sleep()  1 Hz
    
    
    
    if cv.waitKey(20) & 0xFF==ord('s'):
        break
    
  
    
  
    
cv.destroyAllWindows()

if __name__ == '__main__':
    try:
        deteyat()
    #except rospy.ROSInterruptException: pass
        