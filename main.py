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

# initialize dlib's face detector
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args["shape_predictor"])
predictor1 = dlib.shape_predictor("/home/jet/ClionProjects/liandan/cmake-build-debug/sp1.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
capture = cv2.VideoCapture()
capture.open(1)

cv2.namedWindow("TrackingWindow")
data.cap_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
data.cap_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

threading.Thread(target=helpers.sendData).start()

# loop over the frames from the video stream
while True:
    frame = capture.read()[1]
    frame = cv2.resize(frame, (data.cap_width, data.cap_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    face_cnt = len(rects)

    # always use the first face
    if face_cnt > 0:
        points = predictor(gray, rects[0])
        points = helpers.shape_to_np(points)

        cv2.rectangle(frame, (rects[0].left(), rects[0].top()), (rects[0].right(), rects[0].bottom()), (255, 0, 0))

        for (x, y) in points:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        cv2.polylines(frame, numpy.int32([points[:17]]), False, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[17:22]]), False, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[22:27]]), False, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[27:31]]), False, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[31:36]]), False, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[36:42]]), True, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[42:48]]), True, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[48:60]]), True, (0, 0, 255))
        cv2.polylines(frame, numpy.int32([points[60:68]]), True, (0, 0, 255))


        # predictor1
        points = predictor1(gray, rects[0])
        points = helpers.shape_to_np(points)

        for (x, y) in points:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        cv2.polylines(frame, numpy.int32([points[:17]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[17:22]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[22:27]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[27:31]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[31:36]]), False, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[36:42]]), True, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[42:48]]), True, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[48:60]]), True, (0, 255, 0))
        cv2.polylines(frame, numpy.int32([points[60:68]]), True, (0, 255, 0))

        image_points = numpy.array([
            (points[30][0], points[30][1]),  # Nose tip
            (points[8][0], points[8][1]),  # Chin
            (points[5][0], points[5][1]),
            (points[11][0], points[11][1]),
            (points[36][0], points[36][1]),  # Left eye left corner
            (points[39][0], points[39][1]),
            (points[45][0], points[45][1]),  # Right eye right corne
            (points[42][0], points[42][1]),
            (points[48][0], points[48][1]),  # Left Mouth corner
            (points[54][0], points[54][1])  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = numpy.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-300.0, -240.0, -375.0),
            (300.0, -240.0, -375.0),
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (-75.0, 167.0, -135.0),
            (225.0, 170.0, -135.0),  # Right eye right corne
            (75.0, 167.0, -135.0),
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # Camera internals
        focal_length = 720
        center = (data.cap_height / 2, data.cap_width / 2)
        camera_matrix = numpy.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = numpy.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs)

        (nose_end_point2D, jacobian) = cv2.projectPoints(numpy.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)


        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # calculate character data
        data.lock.acquire()
        data.charData = characterCalc.calcAll(points, frame)
        data.lock.release()

    # show the frame
    cv2.imshow("TrackingWindow", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
capture.release()