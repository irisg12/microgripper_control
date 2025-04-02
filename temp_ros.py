#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

from microgripperDetectionNew import microgripperDetection

import cv2 # pip install opencv-python
import numpy as np
import time

def image_callback(msg):
    bridge = CvBridge()
    # queue_size? 
    locating_pub = rospy.Publisher("/microgripper/location", String, queue_size=10)

    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except:
        rospy.logerr("Image could not be read")
        #add code to break
    
    centroids = []
    angles = []
    openColor = (0,0,255) # red
    (tip_xy, openLength, angle), processed_img = microgripperDetection(cv_image, openColor, centroids, angles)
    
    # is pixel sizes fine for this? or need to be converted to real sizes
    locating_msg = f"{tip_xy[0]:.0f} {tip_xy[1]:.0f} {openLength:.3f} {angle:.3f}"
    locating_pub.publish(locating_msg)
    
    if (processed_img is not None):    
        cv2.imshow("Robot Detection", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    else:
        print("No image received")

    if cv2.waitKey(5) & 0xFF == ord(' '): # end video 
        cv2.destroyAllWindows() # need to break from spinning

def main():
    rospy.init_node('image_processor_node', anonymous=True)
    
    rospy.Subscriber("/camera/basler_camera_1/image_raw", Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()


########################

from std_msgs.msg import String

def location_callback(msg):
    data = msg.data.split()
    tip_xy = (float(data[0], float(data[1])))
    openLength = float(data[2])
    angle = float(data[3])
    rospy.loginfo(f"Tip: ({tip_xy[0]}, {tip_xy[1]}), Open Length: {openLength}, Angle: {angle}")
    
def listener():
    rospy.init_node('controller_node', anonymous=True)
    
    rospy.Subscriber("/microgripper/location", String, location_callback)
    rospy.spin()
    
if __name__ == '__main__':
    listener()