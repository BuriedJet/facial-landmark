# import the necessary packages
import threading
import argparse
import cv2
import dlib
import characterCalc
import helpers
import data
import numpy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", type=str, default="./shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

predictor1 = dlib.shape_predictor("/home/jet/ClionProjects/liandan/cmake-build-debug/sp1.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")

cv2.namedWindow("TrackingWindow")

# grab the frame from the threaded video stream, resize it to
# have a maximum width of 400 pixels, and convert it to
# grayscale
frame = cv2.imread("/home/jet/下载/jieke.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
rects = detector(gray, 0)
face_cnt = len(rects)

while True:
    # always use the first face
    for rect in rects:
        # predictor1
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        points = predictor1(gray, rect)
        points = helpers.shape_to_np(points)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in points:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 255), 2)

        cv2.polylines(frame, numpy.int32([points[:17]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[17:22]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[22:27]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[27:31]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[31:36]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[36:42]]), True, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[42:48]]), True, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[48:60]]), True, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[60:68]]), True, (0, 255, 0))

    cv2.imwrite("result2.jpg", frame)

    # show the frame
    cv2.imshow("TrackingWindow", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
