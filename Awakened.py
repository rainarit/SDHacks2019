# Import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FPS
import datetime
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
import csv
import datetime

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

threshold = 0.24
frame_threshold = 20
# Initializing dlib's face detector (HOG-based)
#Detecting the frontal faces
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/shape_predictor_68_face_landmarks.dat')

print("Initializing Facial Landmarking Sensor ->")
# Initializing the camera sensor to warm up
print("Camera Sensor Warming Up ->")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

blink_count = 0

# Looping over all the frames from the webcam stream
total = 0
time = 0
while True:
	ret, frame=vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detector(gray, 0)
	for subject in subjects:
		shape = predictor(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < threshold:  
			blink_count += 1
			if blink_count >= frame_threshold:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				fps = vs.get(cv2.CAP_PROP_FPS)
				time = blink_count/fps
				cv2.putText(frame, "Total Time:" + str(time) + " seconds", (10, 70),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		f = open('/timing.csv','a')
		f.write('\n' + str(datetime.datetime.now()) + ", " + str(time)) #Give your csv text here.
		## Python will convert \n to os.linesep
		f.close()
		cv2.destroyAllWindows()
		vs.release()
		break
