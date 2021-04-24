import cv2 as cv
import numpy as np
import rospy 
from sensor_msgs.msg import CompressedImage
import sys
from duckietown_msgs.msg import WheelsCmdStamped


VERBOSE = False
class image_feature:     
    def __init__(self):
        self.subscriber = rospy.Subscriber("/fatiha/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1)
        #self.published_image = rospy.Publisher("/fatiha/wheels_driver_node/wheels_cmd", WheelsCmdStamped)
        if VERBOSE:
            print("succesfully subscribed to topic")

    def callback(self, ros_data):
        print("We are in the callback")
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        cv.imshow("window",image_np)
        capture = cv.VideoCapture('image_np')

        while True: 
            isTrue, orig_frame = capture.read()

            frame = cv.GaussianBlur(orig_frame, (5, 5), 0)
            hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)



            low_yellow = np.array([18, 94, 140])
            up_yellow = np.array([48, 255, 255])

            #low_green = np.array([40, 40, 45])
            #up_green = np.array([90, 255, 255])

            low_red = np.array([161, 155, 84])
            high_red = np.array([179, 255, 255])


            mask_yellow = cv.inRange(hsv, low_yellow, up_yellow)
            mask_red = cv.inRange(hsv, low_red, high_red)

            edges = cv.Canny(mask_yellow, 75, 150)

            yellow=cv.bitwise_and(frame, frame, mask=mask_yellow)
            red= cv.bitwise_and(frame, frame, mask= mask_red) 

            lines=cv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
            if lines is not None: 
                for line in lines: 
                    x1, y1, x2, y2 = line[0]
                    cv.line(frame, (x1,y1), (x2, y2), (255, 0,0), 5)

            cv.imshow('camera', frame)
            cv.imshow('mask', mask_yellow)
            cv.imshow('red', red)
            cv.imshow('edges', edges)

            
            msg.format = "jpeg"
            msg.data = np.array(cv.imencode('.jpg', image_np)[1]).tostring()
            
            
            self.image_pub.publish(msg)
            #if cv.waitKey(20) & 0xFF==ord('s'):
            #   break
            
        #capture.release()
        #cv.destroyAllWindows()
def main(args):
    ic = image_feature()
    rospy.init_node('line_detect', anonymous= True)
    try: 
        rospy.spin()
    except KeyboardInterrupt:
        print ("Error")
    cv.destroyAllWindows()

if __name__ == '__main__':
    print("WE ARE HERE")
    main(sys.argv)



#create a transformation matrix 
#M = cv.getPerspectiveTransform(src, dst)
#apply the transform to the original image 
#warped = cv.warPerspective(img, M, img_size)
#radius_curvature
#curve_rad = (( 1 +(2*fit[0]*y_eval + fit[1]**2)**1.5)/np.absolute(2*fit[0]))