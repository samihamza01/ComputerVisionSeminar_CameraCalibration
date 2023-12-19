import numpy as np
import cv2 as cv
import os
import camera_model as cm
import matplotlib.pyplot as plt
import time

# Checkerboard square size
SQUARE_SIZE = 29 # mm

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*SQUARE_SIZE
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = os.listdir("C:/Program Files/MATLAB/R2022b/toolbox/vision/visiondata/calibration/slr")
for fname in images:
	img = cv.imread("C:/Program Files/MATLAB/R2022b/toolbox/vision/visiondata/calibration/slr/"+fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, (9,6), None)
	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners2)
		# Draw and display the corners
		cv.drawChessboardCorners(img, (9,6), corners2, ret)
		img = cv.resize(img,(int(img.shape[1]*0.4),int(img.shape[0]*0.4)))
		#cv.imshow('img', img)
		#cv.waitKey(0)
cv.destroyAllWindows()


# Calibrate with OpenCV
RMSErrorCV, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# Calibrate with own implementation
cam = cm.CameraModel()
RMSError, rVecs, tVecs = cam.calibrate(objpoints,imgpoints,gray.shape[0:1])
cam.save_cam_to_json("test.json")

print("Comparison OpenCV and Own Implementation")
print("RMS-Reprojection-Error")
print(f"OpenCV:\t\t{RMSErrorCV}")
print(f"OwnImpl:\t{RMSError}")
print("Camera Matrix")
print(f"OpenCV:\t\t{mtx}")
print(f"OwnImpl:\t{cam.cameraMat}")
print("Distortion Coefficients")
print(f"OpenCV:\t\tk1={dist[0][0]}, k2={dist[0][1]}, k3={dist[0][4]}, p1={dist[0][2]}, p2={dist[0][3]}")
print(f"OwnImpl:\tk1={cam.distortionVec[0]}, k2={cam.distortionVec[1]}, k3={cam.distortionVec[2]}, p1={cam.distortionVec[3]}, p2={cam.distortionVec[4]}")


def comp_params(object_points, image_points, camera_matrix = None, r_vecs = None, t_vecs = None):
	# Calibrate with OpenCV
	print("Starting calibration with OpenCV:")
	startTime = time.time()
	RMSErrorCV, mtxCV, distCV, rvecsCV, tvecsCV = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, r_vecs, t_vecs)
	stopTime = time.time()
	print(f"Calibration took {stopTime-startTime} s.")

	# Calibrate with own implementation
	print("Starting calibration with OpenCV:")
	startTime = time.time()
	cam = cm.CameraModel()
	RMSError, rVecs, tVecs = cam.calibrate(objpoints,imgpoints,gray.shape[0:1])
	stopTime = time.time()
	print(f"Calibration took {stopTime-startTime} s.")








