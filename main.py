# import the necessary packages
import threading
import argparse
import cv2
import dlib
import characterCalc
import helpers
import data

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

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
capture = cv2.VideoCapture()
capture.open(0)

cv2.namedWindow("TrackingWindow")
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

threading.Thread(target=helpers.sendData).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = capture.read()[1]
    frame = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    face_cnt = len(rects)

    # always use the first face
    if face_cnt > 0:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        points = predictor(gray, rects[0])
        points = helpers.shape_to_np(points)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # calculate character data
        data.lock.acquire()
        data.charData = characterCalc.calcAll(points, frame)
        data.lock.release()

    # show the frame
    cv2.imshow("TrackingWindow", frame)
    data.lock.acquire()
    print(data.charData)
    data.lock.release()

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
capture.release()