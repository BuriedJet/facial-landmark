# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	size=frame.shape
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		#print(shape[8][0])
		
		#2D image points. If you change the image, you need to change vector
		image_points = numpy.array([
		(shape[30][0], shape[30][1]),     # Nose tip
		(shape[8][0], shape[8][1]),     # Chin
		(shape[36][0], shape[36][1]),     # Left eye left corner
		(shape[45][0], shape[45][1]),     # Right eye right corne
		(shape[48][0], shape[48][1]),     # Left Mouth corner
		(shape[54][0], shape[54][1])      # Right mouth corner
		], dtype="double")

		#print ("image_points :\n {0}".format(image_points));

		# 3D model points.
		model_points = numpy.array([
		(0.0, 0.0, 0.0),             # Nose tip
		(0.0, -330.0, -65.0),        # Chin
		(-225.0, 170.0, -135.0),     # Left eye left corner
		(225.0, 170.0, -135.0),      # Right eye right corne
		(-150.0, -150.0, -125.0),    # Left Mouth corner
		(150.0, -150.0, -125.0)      # Right mouth corner

		])

		# Camera internals

		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = numpy.array(
			[[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1]], dtype = "double"
			)

		print ("Camera Matrix :\n {0}".format(camera_matrix));

		dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion

		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

		#cv2.cvRodrigues2(rotation_vector,rotation_matrix,0)

		print ("Rotation Vector:\n {0}".format(rotation_vector))
		#print ("Rotation Matrix:\n {0}".format(rotation_matrix))
		print ("Translation Vector:\n {0}".format(translation_vector))

		(nose_end_point2D, jacobian) = cv2.projectPoints(numpy.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

		for p in image_points:
			cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


		p1 = ( int(image_points[0][0]), int(image_points[0][1]))
		p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

		cv2.line(frame, p1, p2, (255, 0, 0), 2)


		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		#for (x, y) in shape:
		#	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
