import cv2 # pip install opencv-python
import numpy as np
import time

def microgripperDetection(color, openColor, centroids, angles):
    img_height = color.shape[0]
    img_width = color.shape[1]
    kernel = np.ones((3,3))
    SEARCH_AREA = img_height/3 
    ASPECTRATIO = 1.4
    NUM_LARGEST = 6 # number of the largest contours to keep/check if robot
    cropping = False 
    bot_hull = None
    tip_avg = [-1.0, -1.0]
    openLength = -1.0
    theta = -180.0
    max_dist = -1
    
    start_time = time.time()

    #color = cv2.resize(color, (1280, 720), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    # Adjust picture contrast / equalize it? 
    #blurred = cv2.bilateralFilter(frame,5,10,10)
    edges = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,135,-5)
    
    edges = cv2.erode(edges, kernel)
    edges = cv2.dilate(edges, kernel, iterations=5)
    edges = cv2.erode(edges, kernel, iterations=3)
    #edges = cv2.dilate(edges, kernel, iterations=1)
    
    #edges =   cv2.bitwise_or(edges, edges_orig)
    
    if cropping and (len(centroids) >= 5):
        crop_mask = np.zeros_like(edges)
        (avg_cx, avg_cy) = np.mean(centroids, axis=0)  
        cv2.rectangle(crop_mask, (int(avg_cx-SEARCH_AREA), int(avg_cy-SEARCH_AREA)), (int(avg_cx+SEARCH_AREA), int(avg_cy+SEARCH_AREA)), 255, thickness=-1)
        edges = cv2.bitwise_and(edges, crop_mask)

    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = list(all_contours)
    if all_contours:
        #! error check contours size
        #sortedPairs = sorted(zip(contours, hierarchy[0]), key=lambda pair: cv2.contourArea(pair[0]), reverse=True)[:6]
        #contours, hierarchy = zip(*(sortedPairs)) # adjust number of accepted contours (10)
        contours = []
        for k in range(min(len(all_contours),NUM_LARGEST)):
            max_idx = max(range(len(all_contours)), key=lambda i: cv2.contourArea(all_contours[i]))
            contours.append(all_contours.pop(max_idx))
            
        for i in range(0, len(contours)):
            hull = cv2.convexHull(contours[i])
            simple_hull = cv2.approxPolyDP(hull, 0.013 * cv2.arcLength(hull, True), True)
            if len(simple_hull) >= 5 and len(simple_hull) < 10: 
                # cur_cnt = cv2.approxPolyDP(contours[i], 0.03 * cv2.arcLength(contours[i], True), True)
                rect = cv2.minAreaRect(contours[i])
                (cx, cy), (width, height), angle = rect  
                #angle = angle+90
                if width < height:
                    width, height = height, width
                    #print("new width:", width)
                else:
                    angle = angle-90 
                if width > img_width/3:
                    continue
                #cv2.drawContours(color, [cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)], 0, openColor, 2)
                if width < ASPECTRATIO*height: # adjust from 1.5
                    if len(centroids) < 5:
                        bot_hull = simple_hull
                        centroids.append((cx,cy))
                        angles.append(angle)
                        #print(rect)
                        break
                    else:
                        (avg_cx, avg_cy) = np.mean(centroids, axis=0)  
                        avg_angle = np.mean(angles)
                        if (True): #(abs(avg_cx-cx) < 30) and (abs(avg_cy-cy) < 30) and (abs(avg_angle - angle) < 40): #! Add if center moves too far 
                            # 30 and 40
                            bot_hull = simple_hull
                            centroids.append((cx,cy))
                            angles.append(angle)

                            if len(centroids) > 5:
                                centroids.pop(0)
                                angles.pop(0)   
                        break                     
                    
        #field = cv2.approxPolyDP(contours[0], 0.11 * cv2.arcLength(contours[0], True), True)    # add this in maybe 
        
        if bot_hull is not None:   
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
            #maxLength = np.max(lengths)
            
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
            tip_avg = (tipl + tipr) / 2
            cv2.line(color, tip_avg.astype(int), mid2.astype(int), openColor, 4)
            v = tip_avg - mid2
            theta = np.arctan2(v[0], v[1])
            theta = np.degrees(theta)
            
            #box = cv2.boxPoints(rect)
            #box = np.intp(box)
            #cv2.drawContours(color, [box], 0, openColor, 2)                    
            
            if len(tipl) != 0:  
                openLength = np.linalg.norm(tipl-tipr) 
                    
                cv2.circle(color, tipl, radius = 6, color=openColor, thickness= -1)
                cv2.circle(color, tipr, radius = 6, color=openColor, thickness= -1)
                cv2.drawContours(color, [simple_hull], 0, openColor, 2)
    else:
        print("No robot contours found.")
    
    #cv2.imshow("Video", color)
    cv2.imshow("Edges", cv2.resize(edges, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    #cv2.imshow("Histogram", equ)
    end_time = time.time()
    #print((end_time-start_time)*1000)
    return ((tip_avg, openLength, theta), max_dist, color)

def spheroidDetection(color, width):
    kernel = np.ones((3,3))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    if width == -1:
        min = 17
        max = 28
    else:
        min = int(width/12)
        max = int(width/9)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=min, maxRadius=max)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(color, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(color, center, radius, (255, 0, 255), 3)
    
def main():
    MS = 50 # milliseconds - 20fps (+ 30 to process each frame)
    centroids = []
    angles = []
    openColor = (0,0,255) # red
    j = 0
    
    vid = cv2.VideoCapture('../new_vid1.mp4') # testVid1.avi
    if not vid.isOpened():
        print("File could not be opened")
    
    while vid.isOpened():
        ret, cv_image = vid.read()
        if not ret:
            break
        
        j += 1    
        (tip_xy, openLength, angle), width, processed_img = microgripperDetection(cv_image, openColor, centroids, angles)
        print(f" {tip_xy[0]:4.0f} {tip_xy[1]:4.0f} {openLength:6.2f} {angle:7.2f}")
        spheroidDetection(cv_image, width)
        if (processed_img is not None):    
            cv2.imshow("Video", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
        else:
            print("No image received")
            
        if cv2.waitKey(MS) & 0xFF == ord(' '):	# end video 
            break
    
    vid.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

