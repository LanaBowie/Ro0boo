from picamera2 import Picamera2
import cv2
import numpy as np


### CONFIGURATION STEP ###
# Cam
CAMERA_ID = 0 ######### ! to be changed for Pi cam

# Box detection by color
BOX_COLOR_LOWER = np.array([0, 180, 180]) ###### red for now
BOX_COLOR_UPPER = np.array([10, 255, 255])
BOX_MIN_AREA = 1000
BOX_ASPECT_MIN = 0.5
BOX_ASPECT_MAX = 2.0

# Aruco
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Display
SHOW_MASK = True
SHOW_DISTANCE = True

### CONFIGURATION STEP - SEE ABOVE ###

# Initialise pi camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Aruco dictionary:
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()

print("Camera work in progress. Press 'q' to quit." )

while True:
    ret, frame = picam2.capture_array()
    
    #Convert picam RGB to OpenCv BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Box detection
    box_mask = cv2.inRange(hsv, BOX_COLOR_LOWER, BOX_COLOR_UPPER)
    box_contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_position = None

    for contour in box_contours:
        area = cv2.contourArea(contour)
        #print(f"Countour area: {area}")

        if area > BOX_MIN_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            #print(f"Drawing box at: ({x}, {y}, {w}, {h})")

            #Filter by shape
            aspect_ratio = w/float(h)
            if BOX_ASPECT_MIN < aspect_ratio < BOX_ASPECT_MAX:
                #Draw green box around:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                #Calculate center
                cx = x + w//2
                cy = y + h//2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                #Label
                cv2.putText(frame, "BOX", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                box_position = (cx, cy)
                print(f"Box detected at: ({cx}, {cy})")

    #Target detection
    # Detect Aruco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    target_position = None

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Get center of first marker
        if len(corners) > 0:
            corner = corners[0][0]
            tx = int(corner[:, 0].mean())
            ty = int(corner[:, 1].mean())

            #draw blue dot at the center:
            cv2.circle(frame, (tx, ty), 5, (255, 0, 0), -1)
            #print(f"Target at ({tx}, {ty})")

            #Label
            cv2.putText(frame, "TARGET", (tx-30, ty-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            target_position = (tx, ty)
            print(f"Target detected at ({tx}, {ty})")

    # Calculate distance (!both detected!) and draw line between
    if box_position and target_position and SHOW_DISTANCE:
        cv2.line(frame, box_position, target_position, (0, 255, 255), 2)

        # Calculate pixel distance
        dx = target_position[0] - box_position[0]
        dy = target_position[1] - box_position[1]
        distance = int(np.sqrt(dx**2 + dy**2))

        # Display distance
        mid_x = (box_position[0] + target_position[0]) // 2
        mid_y = (box_position[1] + target_position[1]) // 2
        cv2.putText(frame, f"{distance}px", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        print(f"Distance: {distance} pixels")

    # Show results
    cv2.imshow('Box and Target Detection', frame)

    if SHOW_MASK:
        cv2.imshow('Mask', box_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
