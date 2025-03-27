#!/usr/bin/env python

#general imports
import math
import numpy as np
import time

#ROS imports
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion  # For robot position
from mag_msgs.msg import DipoleGradientStamped
from std_msgs.msg import Float32MultiArray
import tf.transformations as tf_transfrormations
from sensor_msgs.msg import Image 

#imaging imports
from cv_bridge import CvBridge
import cv2 # pip install opencv-python



def microgripperDetection(color, openColor, centroids, angles, areas, openlengths):

    kernel = np.ones((3,3))
    SEARCH_AREA = 175
    cropping = False
    bot_rect = None
    j = 0
    area_threshold = 0.20  # 20% threshold for contour area change
    
    # Position and angle outlier rejection thresholds
    position_threshold = 3.0  # Standard deviations
    angle_threshold = 3.0     # Standard deviations
    
    # 14umTest1 works
    # 14umTest3 works
    
    start_time = time.time()
    j = j + 1
    
    #color = cv2.resize(color, (1280, 720), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    # Adjust picture contrast / equalize it? 
    #blurred = cv2.bilateralFilter(frame,5,10,10)
    edges = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,135,-6)
    
    edges = cv2.erode(edges, kernel)
    edges = cv2.dilate(edges, kernel, iterations=5)
    edges = cv2.erode(edges, kernel, iterations=3)
    #edges = cv2.dilate(edges, kernel, iterations=1)
    
    #edges = cv2.bitwise_or(edges, edges_orig)
    
    crop_mask = np.zeros_like(edges)
    if cropping:
        cv2.rectangle(crop_mask, (int(cx-SEARCH_AREA), int(cy-SEARCH_AREA)), (int(cx+2*SEARCH_AREA), int(cy+2*SEARCH_AREA)), 255, thickness=-1)
        edges = cv2.bitwise_and(edges, crop_mask)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    current_contour_area = None
    
    if contours:
        #! error check contours size
        sortedPairs = sorted(zip(contours, hierarchy[0]), key=lambda pair: cv2.contourArea(pair[0]), reverse=True)[:6]
        contours, hierarchy = zip(*(sortedPairs)) # adjust number of accepted contours (10)
        for i in range(0, len(contours)):
            hull = cv2.convexHull(contours[i])
            simple_hull = cv2.approxPolyDP(hull, 0.013 * cv2.arcLength(hull, True), True)
            if len(simple_hull) >= 5 and len(simple_hull) < 10: # was for ellipse fitting - speeds up processing but maybe remove
                # cur_cnt = cv2.approxPolyDP(contours[i], 0.03 * cv2.arcLength(contours[i], True), True)
                rect = cv2.minAreaRect(contours[i])
                (cx, cy), (width, height), angle = rect  
                #angle = angle+90
                if width < height:
                    width, height = height, width
                    #print("new width:", width)
                else:
                    angle = angle-90 
                area = cv2.contourArea(contours[i])
                #cv2.drawContours(color, [cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)], 0, openColor, 2)
                if width < 1.75*height and area < 50000 and area > 5000: # adjust from 1.5
                    # Calculate contour area
                    

                    
                    # Perform outlier rejection if we have enough history
                    is_outlier = False
                    if len(centroids) >= 10:
                        # Check if area is within threshold of previous area
                        mean_area = np.mean(areas)  
                        std_area = np.std(areas)
                        if std_area > 0:  # Avoid division by zero
                            # Calculate z-score for area
                            z_area = abs(area - mean_area) / std_area
                            if z_area > 3.0:  # 3 standard deviations
                                print(f"Area outlier rejected: {area:.0f} - z-score: {z_area:.2f}")
                                is_outlier = True

                        # Calculate statistics for centroids
                        centroids_array = np.array(centroids)
                        mean_x = np.mean(centroids_array[:, 0])
                        mean_y = np.mean(centroids_array[:, 1])
                        std_x = np.std(centroids_array[:, 0])
                        std_y = np.std(centroids_array[:, 1])
                        
                        # Check if new position is an outlier
                        if std_x > 0 and std_y > 0:  # Avoid division by zero
                            z_x = abs(cx - mean_x) / std_x
                            z_y = abs(cy - mean_y) / std_y
                            
                            if z_x > position_threshold or z_y > position_threshold:
                                print(f"Position outlier rejected: ({cx:.1f}, {cy:.1f}) - z-scores: x={z_x:.2f}, y={z_y:.2f}")
                                is_outlier = True
                        
                        # Calculate statistics for angles
                        mean_angle = np.mean(angles)
                        std_angle = np.std(angles)
                        
                        # Check if new angle is an outlier (handling circular nature of angles)
                        if std_angle > 0:  # Avoid division by zero
                            # Calculate smallest angle difference accounting for wraparound
                            angle_diff = min(abs(angle - mean_angle), 360 - abs(angle - mean_angle))
                            z_angle = angle_diff / std_angle
                            
                            if z_angle > angle_threshold:
                                print(f"Angle outlier rejected: {angle:.1f}° - z-score: {z_angle:.2f}")
                                is_outlier = True
                    
                    # Skip this contour if it's an outlier
                    if is_outlier:
                        openColor = (0, 0, 255)  # red
                        continue
                    
                    if len(centroids) < 5:
                        bot_rect = rect
                        bot_cnt = contours[i]
                        centroids.append((cx,cy))
                        angles.append(angle)
                        areas.append(area)
                        openColor = (0, 255, 0)  # green
                        #print(rect)
                        break
                    else:
                        #cropping = True
                        (avg_cx, avg_cy) = np.mean(centroids, axis=0)  
                        avg_angle = np.mean(angles)
                        
                        # Accept the detection
                        bot_rect = rect
                        bot_cnt = contours[i]
                        centroids.append((cx,cy))
                        angles.append(angle)
                        areas.append(area)
                        openColor = (0, 255, 0)  # green
                        
                        if len(centroids) > 5:
                            centroids.pop(0)
                            angles.pop(0)
                            areas.pop(0)   
                        break                     
                    
        #field = cv2.approxPolyDP(contours[0], 0.11 * cv2.arcLength(contours[0], True), True)    # add this in maybe 
        
        if bot_rect:   
            thetas = []    
            lengths = []   
            for i in range(len(simple_hull)):
                v1 = (simple_hull[i-1] - simple_hull[i]).flatten()
                v2 = (simple_hull[(i+1) % len(simple_hull)] - simple_hull[i]).flatten()
                theta = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                theta = np.degrees(np.arccos(theta))
                thetas.append((theta,i))
                lengths.append((np.linalg.norm(v1),i))
                
            #print(thetas)
            #print("lens", lengths)
            maxLength = np.max(lengths)
            
            simple_hull = np.array(simple_hull).reshape(-1, 2)
    
            max_dist = 0
            side1, side2 = None, None
            
            for i in range(len(simple_hull)):
                for j in range(i + 1, len(simple_hull)):  # Avoid redundant comparisons
                    dist = np.linalg.norm(simple_hull[i] - simple_hull[j])
                    if dist > max_dist:
                        max_dist = dist
                        side1, side2 = i, j
            
            left = lengths[side1][0]
            right = lengths[(side1+1) % len(lengths)][0] 
            if (left > right):
                tipr = simple_hull[side1-1]  
                tipl = simple_hull[(side2+1) % len(lengths)]  
            else:
                tipl = simple_hull[(side1+1) % len(lengths)]
                tipr = simple_hull[side2-1]  
            
            mid2 = (simple_hull[side1] + simple_hull[side2]) / 2
            mid1 = (tipl + tipr) / 2
            cv2.line(color, mid1.astype(int), mid2.astype(int), openColor, 4)
            v = mid1 - mid2
            angle = np.arctan2(v[0], v[1])
            angle = np.degrees(angle)
            #print("angle", angle)             
            
            if len(tipl) != 0:  
                openlength = np.linalg.norm(tipl-tipr) 
                if len(openlengths)>=5:
                    avg_openlength = np.mean(openlengths) 
                    z_openlength = abs(openlength - avg_openlength) / np.std(openlengths)
                    if z_openlength > 3.0:  # 3 standard deviations
                        openlengths.append(openlength)
                    openlengths.pop(0)

                cv2.circle(color, tipl, radius = 6, color=openColor, thickness= -1)
                cv2.circle(color, tipr, radius = 6, color=openColor, thickness= -1)
                cv2.drawContours(color, [simple_hull], 0, openColor, 2)    
    else:
        print("No robot contours found.")
    
    #cv2.imshow("Video", color)
    cv2.imshow("Edges", edges)
    #cv2.imshow("Histogram", equ)
    end_time = time.time()
    #print((end_time-start_time)*1000)
    return color, openColor, centroids, angles, areas, openlengths

def ProcessVideo():
    MS = 50 # milliseconds - 20fps (+ 30 to process each frame)
    centroids = []
    angles = []
    areas = []
    openColor = (0,0,255) # red
    
    vid = cv2.VideoCapture('../new_vid2.mp4') # testVid1.avi
    if not vid.isOpened():
        print("File could not be opened")
      
    while vid.isOpened():
        ret, cv_image = vid.read()
        if not ret:
            break
    
        processed_img, openColor, centroids, angles, areas, openlength= microgripperDetection(cv_image, openColor, centroids, angles, areas)
        if (processed_img is not None):    
            cv2.imshow("Video", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
        else:
            print("No image received")
            
        if cv2.waitKey(MS) & 0xFF == ord(' '):	# end video 
            break
    
    vid.release()
    cv2.destroyAllWindows()

def publish_pose(publisher,x,y,theta,opening,timestamp=None):
        # Convert the orientation to quaternion (only around z-axis)
        try:  
            # Create message
            pose_msg = Float32MultiArray()
            pose_msg.data = [x, y, theta, opening, timestamp.to_sec()]

            # Publish the message
            publisher.publish(pose_msg)
        except Exception as e:
            print(f"Error publishing pose: {e}")

def image_callback(msg):
    bridge = CvBridge()
    global centroids, angles, areas, publisher, openlengths
    openColor = (0,255,0)  # green
    timestamp = msg.header.stamp
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except:
        rospy.logerr("Image could not be read")
        return
    processed_img, openColor, centroids, angles, areas, openlengths = microgripperDetection(cv_image, openColor, centroids, angles, areas, openlengths)
    if (processed_img is not None):
        publish_pose(publisher, centroids[-1][0], centroids[-1][1], angles[-1], openlengths, timestamp)
        cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
        if cv2.waitKey(2) & 0xFF == ord(' '):	# end video 
            cv2.destroyAllWindows() # need to break from spinning
def main():
    rospy.init_node('image_processor_node', anonymous=True)
    
    # Initialize global variables
    global centroids, angles, prev_contour_area, publisher, areas, openlengths
    openlengths = []
    areas = []
    centroids = []
    angles = []
    prev_contour_area = None
    
    rospy.Subscriber("/camera/basler_camera_1/image_raw", Image, image_callback)
    publisher = rospy.Publisher('/vision_feedback/pose_estimation', Float32MultiArray, queue_size=10)
    rospy.spin()   

if __name__ == "__main__":
    main()

