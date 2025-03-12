import cv2 # pip install opencv-python
import numpy as np
import time

MS = 20 # milliseconds - 20fps (+ 30 to process each frame)
kernel = np.ones((3,3))
centroids = []
angles = []
openColor = (0,0,255) # red
SEARCH_AREA = 175
cropping = False
bot_rect = None

# 14umTest1 works
# 14umTest3 works

vid = cv2.VideoCapture('16umSerpTest1.mp4') # testVid1.avi

if not vid.isOpened():
    print("File could not be opened")
    
#out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280,960))
j = 0
while vid.isOpened(): # and j < 20*30:
    start_time = time.time()
    j = j + 1
    ret, color = vid.read()
    if not ret:
        break
    
    color = cv2.resize(color, (1280, 720), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    # Adjust picture contrast / equalize it? 
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(24,24))
    equ = clahe.apply(frame)
    edges_orig = cv2.Canny(equ,50,150) # adjust: gradient <25 is rejected, in between depends on connectivity
    edges_orig = cv2.dilate(edges_orig, kernel, iterations=3)
    edges_orig = cv2.erode(edges_orig, kernel, iterations=2)
    
    edges = cv2.dilate(edges_orig, kernel, iterations=7)
    edges = cv2.erode(edges, kernel, iterations=7)
    #edges = cv2.dilate(edges, kernel, iterations=1)
    
    edges = cv2.bitwise_or(edges, edges_orig)
    
    crop_mask = np.zeros_like(edges)
    if cropping:
        cv2.rectangle(crop_mask, (int(cx-SEARCH_AREA), int(cy-SEARCH_AREA)), (int(cx+2*SEARCH_AREA), int(cy+2*SEARCH_AREA)), 255, thickness=-1)
        edges = cv2.bitwise_and(edges, crop_mask)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        #! error check contours size
        sortedPairs = sorted(zip(contours, hierarchy[0]), key=lambda pair: cv2.contourArea(pair[0]), reverse=True)[:5]
        contours, hierarchy = zip(*(sortedPairs)) 				# adjust number of accepted contours (10)
        for i in range(0, len(contours)):
            #! fit ellipse requires at least 5 points 
            if len(contours[i]) >= 5:
                # cur_cnt = cv2.approxPolyDP(contours[i], 0.03 * cv2.arcLength(contours[i], True), True)
                rect = cv2.minAreaRect(contours[i])
                (cx, cy), (width, height), angle = rect  
                #angle = angle+90
                if width > height:
                    width, height = height, width
                    #print("new width:", width)
                else:
                    angle = angle-90
                    
                if height < 1.5*width: # adjust
                    #cv2.drawContours(color, [cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)], 0, openColor, 2)
                    if len(centroids) < 5:
                        bot_rect = rect
                        bot_cnt = contours[i]
                        centroids.append((cx,cy))
                        angles.append(angle)
                        #print(rect)
                        break
                    else:
                        #cropping = True
                        (avg_cx, avg_cy) = np.mean(centroids, axis=0)  
                        avg_angle = np.mean(angles)
                        print(avg_cy-cy)
                        if (abs(avg_cx-cx) < 20) and (abs(avg_cy-cy) < 30) and (abs(avg_angle - angle) < 40): #! Add if center moves too far 
                            print("good")
                            """if hierarchy[i][2] == -1: # no child contours = green drawing
                                openColor = (0, 255, 0)
                            else:
                                openColor = (0, 0, 255)"""
                            bot_rect = rect
                            bot_cnt = contours[i]
                            centroids.append((cx,cy))
                            angles.append(angle)

                            if len(centroids) > 5:
                                centroids.pop(0)
                                angles.pop(0)
                        break
                        
                    
        #field = cv2.approxPolyDP(contours[0], 0.11 * cv2.arcLength(contours[0], True), True)    # add this in maybe 
        if bot_rect:
                            
            (cx, cy), (width, height), angle = bot_rect
            if width > height:
                width, height = height, width
                angle = angle+90
            angle = np.radians(angle-90)   
            
            unit_x = np.sin(angle)
            unit_y = np.cos(angle)
            x1, y1 = int(cx + width/2 * unit_x), int(cy - width/2 * unit_y)
            x2, y2 = int(cx - width/2 * unit_x), int(cy + width/2 * unit_y)

            #cv2.ellipse(color,bot_ell, (0,0,255), 2)
            #(cx, cy), (minor, major), angle = bot_ell # angle in degrees
            #angle = np.radians(angle+90)

            # Compute front and back endpoints along the semi-minor axis
            #x1, y1 = int(cx + minor/2 * np.sin(angle)), int(cy - minor/2 * np.cos(angle))
            #x2, y2 = int(cx - minor/2 * np.sin(angle)), int(cy + minor/2 * np.cos(angle))

            # Create a blank image for the minor axis line
            line1 = np.zeros_like(edges)
            line2 = np.zeros_like(edges)

            cv2.line(line1, (x1, y1), (int(cx), int(cy)), 255, 4)
            cv2.line(line2, (x2, y2), (int(cx), int(cy)), 255, 4)

            num_white1 = np.sum(cv2.bitwise_and(edges, line1) > 0) 
            num_white2 = np.sum(cv2.bitwise_and(edges, line2) > 0)
            total_white = np.sum(line1 > 0)
            front = (x2,y2)
            back = (x1,y1)

            # Determine front direction
            if num_white2 > num_white1:
                front = (x1, y1)
                back = (x2,y2)
                percent = float(num_white1) / total_white 
            cv2.line(color, front, (int(cx), int(cy)), openColor, 2)
            
            # back_rect = [(back[0] - unit_x,back[1] - unit_y), (back[0] + unit_x, back[1] + unit_y)]
            Rback_x = cx - width*.375*unit_x + height*.375*unit_y
            Rback_y = cy + width/4.0*unit_y + height/4.0*unit_x
            Rback = ((Rback_x, Rback_y), (height/4.0, height/4.0), np.degrees(angle))
            
            Lback_x = cx - width*.375*unit_x - height*.375*unit_y
            Lback_y = cy + width/4.0*unit_y - height/4.0*unit_x
            Lback = ((Lback_x, Lback_y), (height/4.0, height/4.0), np.degrees(angle))
            
            Rbox = cv2.boxPoints(Rback)
            Rbox = np.intp(Rbox)
            cv2.drawContours(line1, [Rbox], 0, 255, -1)
            
            Lbox = cv2.boxPoints(Lback)
            Lbox = np.intp(Lbox)
            cv2.drawContours(line1, [Lbox], 0, 255, -1)
            
            num_white1 = np.sum(cv2.bitwise_and(edges, line1) > 0) 
            total_white = np.sum(line1 > 0)
            percent = float(num_white1) / total_white 
            if percent < .85:
                openColor = (0, 255, 0)   
            elif percent > .875:
                openColor = (0, 0, 255)   
            box = cv2.boxPoints(bot_rect)
            box = np.intp(box)
            cv2.drawContours(color, [box], 0, openColor, 2)
    
    else:
        print("No robot contours found.")
    
    cv2.imshow("Video", color)
    cv2.imshow("Edges", edges)
    cv2.imshow("Histogram", equ)
    #out.write(frame)
    end_time = time.time()
    #print((end_time-start_time)*1000)
    if cv2.waitKey(MS) & 0xFF == ord(' '):	# end video 
        break
    
vid.release()
#out.release()
cv2.destroyAllWindows()
