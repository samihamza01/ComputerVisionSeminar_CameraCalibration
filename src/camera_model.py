"""@package CameraModel
Author: Sami Hamza

Python implementation of a camera model.
"""

import numpy as np
import json
import levenberg_marquardt as lm
import numpy.typing as npt
import typing as tp
import matplotlib.pyplot as plt
import homography as hom

class CameraModel():
	"""Camera model according to Zhang.
	"""
	# camera attributes
	cameraMat: npt.ArrayLike
	distortionVec: npt.ArrayLike
	imageSize: tp.Tuple[int,int]

	# calibration specific attributes
	calibrationDataStructure: list
	residualParamBuffer: npt.ArrayLike
	residualBuffer: npt.ArrayLike

	
	def __init__(self, cameraMat: tp.Optional[npt.ArrayLike] = None, distortionVec: tp.Optional[npt.ArrayLike] = None, imageSize: tp.Optional[tp.Tuple[int,int]] = None) -> None:
		"""Constructor.

		Args:
			cameraMat (tp.Optional[npt.ArrayLike]): 		Camera Matrix of the camera object. If none is given, it is set to a 3x3 matrix of zeros.
			distortionVec (tp.Optional[npt.ArrayLike]): 	Distortion vector of the camera object containing the 3 radial and the 2 tangential coefficients.
															If none is given, it is set to a 5x1 vector of zeros.
			imageSize (tp.Optional[npt.ArrayLike] = None):	Size of the images used for calibration as vector of integers in x and y direction.

		"""
		if cameraMat is None:
			self.cameraMat = np.zeros((3,3),dtype=float)
		else:
			self.cameraMat = cameraMat

		if distortionVec is None:
			self.distortionVec = np.zeros((17,),dtype=float)
		else:
			self.distortionVec = distortionVec

		if imageSize is None:
			self.imageSize = (0,0)
		else:
			self.imageSize = imageSize

		return

	def world_2_image(self, objectPoint: npt.ArrayLike,rVec: npt.ArrayLike, tVec: npt.ArrayLike) -> npt.ArrayLike:
		"""Function to project an object point with given orientation and position to camera image plane.

		Args:
			objectPoint (npt.ArrayLike): 3D Point to project.
			rVec (npt.ArrayLike): Vector specifing the rotation axis and angle (norm(rotVec)) of the pose of the camera in the reference system (the pattern system).
			tVec (npt.ArrayLike): Vector specifing the translation of th pose of the camera in the reference system (the pattern system).

		Returns:
			npt.ArrayLike: The 2D point in the image plane of the camera.
		"""
		# Transform from World to Cameracoordinates
		if np.linalg.norm(rVec) != 0:
			camPoint = self._rot_around_axis(objectPoint,rVec/np.linalg.norm(rVec),-np.linalg.norm(rVec)) + tVec
		else:
			camPoint = objectPoint + tVec

		# Project in normalised image plane
		normImgPoint = np.array(	[camPoint[0]/camPoint[2],
									camPoint[1]/camPoint[2],
									1])

		# Add distortion
		# Radius
		r_square = normImgPoint[0]**2+normImgPoint[1]**2

		# radial component
		distRadX = normImgPoint[0]*(1 + self.distortionVec[0]*r_square + self.distortionVec[1]*r_square**2 + self.distortionVec[2]*r_square**3)
		distRadY = normImgPoint[1]*(1 + self.distortionVec[0]*r_square + self.distortionVec[1]*r_square**2 + self.distortionVec[2]*r_square**3)

		# tangential component
		distTanX = 2*self.distortionVec[3]*normImgPoint[0]*normImgPoint[1] + self.distortionVec[4]*(r_square + 2*normImgPoint[0]**2)
		distTanY = 2*self.distortionVec[4]*normImgPoint[0]*normImgPoint[1] + self.distortionVec[3]*(r_square + 2*normImgPoint[1]**2)

		normDistImgPoint = np.array([	distRadX + distTanX,
										distRadY + distTanY,
										1])

		# Project in image plane
		imgPoint = np.matmul(self.cameraMat,normDistImgPoint)

		return imgPoint[0:1]

	def _rot_around_axis(self, vector: npt.ArrayLike, axis: npt.ArrayLike, angle: float) -> npt.ArrayLike:
		"""Helper function to rotate a vector around the normed axis n by the specified angle.

		Args:
			vector (npt.ArrayLike): 3D vector Vector to rotate.
			axis (npt.ArrayLike): 3D normed axis of rotation.
			angle (float): Angle to rotate in rad. The sign is defined by the right hand rule.

		Returns:
			npt.ArrayLike: Rotated vector.
		"""

		# perform a rotation according to rodrigues
		# v' = v*cos(phi)+(axis x v)*sin(phi)+axis * (axis * v)*(1-cos(phi))
		# with phi = norm(axis)

		# as matrix (Rotationmatrix) => faster
		# reuse the cos and sin
		cAng = np.cos(angle)
		sAng = np.sin(angle)
		# reuse the value of 1-cos (shifted cos)
		s_CAng = 1 - cAng

		rotMat = np.array([	[cAng + axis[0]**2*s_CAng, axis[0]*axis[1]*s_CAng - axis[2]*sAng, axis[0]*axis[2]*s_CAng + axis[1]*sAng],
							[axis[0]*axis[1]*s_CAng + axis[2]*sAng, cAng + axis[1]**2*s_CAng, axis[1]*axis[2]*s_CAng - axis[0]*sAng],
							[axis[0]*axis[2]*s_CAng - axis[1]*sAng, axis[1]*axis[2]*s_CAng + axis[0]*sAng, cAng + axis[2]**2*s_CAng]])

		v_rot = np.matmul(rotMat, vector)
		return v_rot

	def calibrate(self, objectPoints: tp.List[tp.List[npt.ArrayLike]], imagePoints: tp.List[tp.List[npt.ArrayLike]], imageSize: tp.Tuple[int,int], cameraMatrix: tp.Optional[npt.ArrayLike] = None, rVecs: tp.Optional[tp.List[npt.ArrayLike]] = None, tVecs: tp.Optional[tp.List[npt.ArrayLike]] = None, verbose: tp.Optional[bool] = False) -> tp.Tuple[float,tp.List[npt.ArrayLike],tp.List[npt.ArrayLike]]:
		"""_summary_

		Args:
			objectPoints (tp.List[tp.List[npt.ArrayLike]]): List of List of Object Points for the corresponding views.
			imagePoints (tp.List[tp.List[npt.ArrayLike]]): List of List of Image Points for the corresponding views.
			imageSize (tp.Tuple[int,int]): Size of the images used for calibration.
			cameraMatrix (tp.Optional[npt.ArrayLike], optional): Initial guess for camera matrix. Defaults to None.
			rVecs (tp.Optional[tp.List[npt.ArrayLike]], optional): Initial guess for rotation vectors. Defaults to None.
			tVecs (tp.Optional[tp.List[npt.ArrayLike]], optional): Initial guess for translation vectors. Defaults to None.
			verbose (tp.Optional[bool], optional): Displays some information about the optimization. Defaults to False.

		Raises:
			ValueError: If nonesense input.
			ValueError: If nonesense input.
			ValueError: If nonesense input.

		Returns:
			tp.Tuple[float,tp.List[npt.ArrayLike],tp.List[npt.ArrayLike]]: 	Tuple of the RMS-Reprojection-Error, \
																			list of the estimated roatation vectors, \
																			list of the estimated translation vectors.														
		"""

		# Perform some input parameter checks
		if rVecs is None and tVecs is None and cameraMatrix is None:
			estimateInitGuess = True
		elif rVecs is not None and tVecs is not None and cameraMatrix is not None:
			estimateInitGuess = False
			if len(objectPoints) != len(rVecs) or len(objectPoints) != len(tVecs):
				raise ValueError("Translation and rotation vectors must have same length as the number of views (images).")
			if cameraMatrix.shape != (3,3):
				raise ValueError("Camera matrix must be of shape (3,3)")
		else:
			raise ValueError("If giving an initial guess, the camera matrix, rotation vectors and translation vectors must be given.")
		
		# get calibration data structure and length
		self.calibrationDataStructure = []
		length = 0
		for view in objectPoints:
			self.calibrationDataStructure.append(len(view))
			length += len(view)
		
		numViews = len(objectPoints)
		parameterVec = np.zeros((9+6*numViews,),dtype=np.float64)

		# Initial Guess (given or estimated)
		if estimateInitGuess:
			# Estiamation via closed form solution
			homographies = []
			for viewIdx, objectPointsView in enumerate(objectPoints):
				homography = hom.compute_homography(objectPoints[viewIdx], imagePoints[viewIdx])
				homographies.append(homography)
			# set camera matrix
			cameraMatrixInit = hom.estimate_overall_camera_matrix(homographies, imageSize)
			parameterVec[0] = cameraMatrixInit[0,0]
			parameterVec[1] = cameraMatrixInit[1,1]
			parameterVec[2] = cameraMatrixInit[0,2]
			parameterVec[3] = cameraMatrixInit[1,2]

			# set roations and translation parameters
			rVecs, tVecs = hom.extract_rotation_translation(cameraMatrixInit,homographies)
			t_offset = len(objectPoints)*3 + 9
			for viewIdx in range(0,len(objectPoints)):
				parameterVec[9+viewIdx*3:12+viewIdx*3] = rVecs[viewIdx][:].reshape((3,))
				parameterVec[t_offset+viewIdx*3:t_offset+3+viewIdx*3] = tVecs[viewIdx][:].reshape((3,))
		else:
			parameterVec[0] = cameraMatrix[0,0]
			parameterVec[1] = cameraMatrix[1,1]
			parameterVec[2] = cameraMatrix[0,2]
			parameterVec[3] = cameraMatrix[1,2]
			
			t_offset = len(objectPoints)*3 + 9
			for viewIdx in range(0,len(objectPoints)):
				parameterVec[9+viewIdx*3:12+viewIdx*3] = rVecs[viewIdx][:]
				parameterVec[t_offset+viewIdx*3:t_offset+3+viewIdx*3] = tVecs[viewIdx][:]
		
		# Optimization
		# create optimzer
		levMar = lm.LevenbergMarquardtOptimizer(maxIterations=35,parameterStepThr=5e-3,gradientThr=1e-6)

		# serialize the object and image data
		objectPointsSer = np.zeros((length,3))
		imagePointsSer = np.zeros((length,2))
		serCounter = 0
		for viewIdx, pointsPerView in enumerate(self.calibrationDataStructure):
			for pointIdx in range(0,pointsPerView):
				objectPointsSer[serCounter][:] = objectPoints[viewIdx][pointIdx]
				imagePointsSer[serCounter][:] = imagePoints[viewIdx][pointIdx]
				serCounter += 1

		# call _residual_function with initCall=True befor optimization
		self._residual_function(parameterVec,objectPointsSer,imagePointsSer,initCall=True)
		optimalParams, final_square_error, _, iteration, conv, squareErrorHist = levMar.optimize(self._residual_function,parameterVec,objectPointsSer,imagePointsSer)

		self.cameraMat = np.array([	[optimalParams[0],0,optimalParams[2]],
									[0,optimalParams[1],optimalParams[3]],
									[0,0,1]])
		self.distortionVec = optimalParams[4:9]
		self.imageSize = imageSize
		RMSError = np.sqrt(final_square_error/length)

		# get rVecs and tVecs
		rVecs = []
		tVecs = []
		t_offset = len(objectPoints)*3+9
		for viewIdx in range(0,len(objectPoints)):
			rVecs.append(parameterVec[9+viewIdx*3:12+viewIdx*3])
			tVecs.append(parameterVec[t_offset+viewIdx*3:t_offset+3+viewIdx*3])

		if verbose:
			# show result
			if conv == 0:
				print(f"Convergence in gradient after {iteration} iterations.")
			elif conv == 1:
				print(f"Convergence in parameters after {iteration} iterations.")
			else:
				print(f"Max iterations reached [{iteration}]!")
			print(f"Final RMS reprojection error: {RMSError}")
			# plot history
			plt.figure()
			ax = plt.subplot(111)
			ax.plot(np.linspace(0,iteration,iteration+1),squareErrorHist)
			ax.set_title("Convergence Plot")
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Square Error")
			ax.set_xlim(0,iteration)
			ax.set_ylim(bottom=0)
			plt.show()
		return (RMSError, rVecs, tVecs)
	
	def _world_2_image(self, parameterVec: npt.ArrayLike, objectPoint: npt.ArrayLike) -> npt.ArrayLike:
		"""Helper function to transform an object/world point to the image plane of the camera.

		Args:
			parameterVec (npt.ArrayLike): Parameter vector containing (alpha,beta,u0,v0,k1,k2,k3,p1,p2,r1,r2,r3,t1,t2,t3). In this parameter vecotr,\
											only the extrinsics for this particular view are listed.
			objectPoint (npt.ArrayLike): 3D object/world point to transform to the image plane.

		Returns:
			npt.ArrayLike: The 2D point in the image plane.
		"""
		# Transform from World to Cameracoordinates
		if np.linalg.norm(parameterVec[9:12]) != 0:
			camPoint = self._rot_around_axis(objectPoint,parameterVec[9:12]/np.linalg.norm(parameterVec[9:12]),np.linalg.norm(parameterVec[9:12]))+parameterVec[12:15]
		else:
			camPoint = objectPoint + parameterVec[12:15]

		# Project in normalised image plane
		normImgPoint = np.array(	[camPoint[0]/camPoint[2],
									camPoint[1]/camPoint[2],
									1])

		# Add distortion
		# Radius
		r_square = normImgPoint[0]**2+normImgPoint[1]**2

		# radial component
		distRadX = normImgPoint[0]*(1 + parameterVec[4]*r_square + parameterVec[5]*r_square**2 + parameterVec[6]*r_square**3)
		distRadY = normImgPoint[1]*(1 + parameterVec[4]*r_square + parameterVec[5]*r_square**2 + parameterVec[6]*r_square**3)

		# tangential component
		distTanX = 2*parameterVec[7]*normImgPoint[0]*normImgPoint[1] + parameterVec[8]*(r_square + 2*normImgPoint[0]**2)
		distTanY = 2*parameterVec[8]*normImgPoint[0]*normImgPoint[1] + parameterVec[7]*(r_square + 2*normImgPoint[1]**2)

		normDistImgPoint = np.array([	distRadX + distTanX,
										distRadY + distTanY,
										1])

		# Project in image plane
		camMat = np.array([	[parameterVec[0],0,parameterVec[2]],
							[0,parameterVec[1],parameterVec[3]],
							[0,0,1]])
		imgPoint = np.matmul(camMat[0:2,:],normDistImgPoint)

		return imgPoint

	def _residual_function(self, parameterVec: npt.ArrayLike, objectPoints: npt.ArrayLike, imagePoints: npt.ArrayLike, initCall: tp.Optional[bool] = False) -> npt.ArrayLike:
		"""Function calculating the residual vector for optimization with Levenberg Marquardt. Must be called with `initCall = True` before first use in calibration.

		Args:
			parameterVec (npt.ArrayLike): Parameter vector containing (alpha,beta,u0,v0,k1,k2,k3,p1,p2,r1,r2,r3,...,t1,t2,t3,...).
			objectPoints (npt.ArrayLike): Object point list, where the outer list contains the view sets and the inner list contains the points per view.
			imagePoints (npt.ArrayLike): Image point list, where the outer list contains the view sets and the inner list contains the points per view. 
			initCall (tp.Optional[bool]): indicating if it is the first call in the calibration.

		Returns:
			npt.ArrayLike: Residual Vector.
		"""
		# determine size of residual vector and create it
		if initCall==True:
			length = objectPoints.shape[0]
			residualVec = np.zeros((length,),dtype=np.float64)
		else:
			residualVec = self.residualBuffer.copy()

		# try to find a way to reduce calculations in jacobian
		# only compute full residual vector, if:
		# 1) camera matrix
		# 2) dist coeff have changed
		# else compute only the view where params have changed
		
		if not initCall and (parameterVec[0:9]==self.residualParamBuffer[0:9]).all():
			computeAll = False
		else:
			computeAll = True

		# parameter vector
		parameterViewVector = np.zeros((15,),dtype=np.float64)
		parameterViewVector[0:9] = parameterVec[0:9]

		# calculate residual vector
		paramOffset =  len(self.calibrationDataStructure)*3 + 8

		viewOffset = 0
		for idxView, elementsPerView in enumerate(self.calibrationDataStructure):
			# check if current view params have changed
			if not computeAll and ((parameterVec[8+3*idxView+1:11+idxView*3+1]==self.residualParamBuffer[8+3*idxView+1:11+idxView*3+1]).all() and (parameterVec[paramOffset+3*+idxView+1:paramOffset+3+3*idxView+1]==self.residualParamBuffer[paramOffset+3*+idxView+1:paramOffset+3+3*idxView+1]).all()):
				viewOffset += elementsPerView 
				continue
			# get current view params
			parameterViewVector[9:12] =  parameterVec[8+3*idxView+1:11+idxView*3+1]
			parameterViewVector[12:15] =  parameterVec[paramOffset+3*+idxView+1:paramOffset+3+3*idxView+1]
			for idxPoint in range(0,elementsPerView):
				estimatedPoint = self._world_2_image(parameterViewVector,objectPoints[idxPoint+viewOffset])
				residualVec[idxPoint+viewOffset] = np.linalg.norm(imagePoints[idxPoint+viewOffset] - estimatedPoint)
			viewOffset += elementsPerView 

		self.residualBuffer = residualVec.copy()
		self.residualParamBuffer = parameterVec.copy()
		return residualVec

	def save_cam_to_json(self, path: str):
		"""Function to save the camera matrix and the distortion vector to a json file.

		Args:
			path (str): Realtive or absolute path and filename to store the file in (e.g. ~/camData.json).
		"""
		camData = {"cameraMat": self.cameraMat.tolist(), "distortionVec": self.distortionVec.tolist(), "imageSize": list(self.imageSize)}
		with open(path,'w') as f:
			json.dump(camData,f)
		return

	def load_cam_from_json(self, path: str):
		"""Function to load the camera matrix and the distortion vector from a json file.

		Args:
			path (str): Realtive or absolute path and filename to load the file in (e.g. ~/camData.json).
		"""
		with open(path,'r') as f:
			camData = json.load(f)
		self.cameraMat = np.array(camData["cameraMat"])
		self.distortionVec = np.array(camData["distortionVec"])
		self.imageSize = tuple(camData["imageSize"])
		return

	def undistort_image(self, image: npt.NDArray) -> npt.NDArray:
		"""Undistort a given image. If the image size is different from the image size used for calibration, then the cameramatrix is automatically scaled.

		Args:
			image (npt.NDArray): Image to undistort.

		Returns:
			npt.NDArray: Undistorted image.
		"""
		pix_v, pix_u, channels = image.shape

		camera_matrix = self.cameraMat.copy()
		if (pix_v, pix_u) != self.imageSize:
			# rescale camera matrix
			camera_matrix[0,:] *= pix_u/self.imageSize[1]
			camera_matrix[1,:] *= pix_v/self.imageSize[0]

		pixel_coords_uv=np.ones((3,pix_u*pix_v))
		pixel_coords_uv[0:2,:] = np.mgrid[0:pix_u,0:pix_v].reshape(2,-1)

		# project image points to normalised image plane
		inv_camera_matrix = np.linalg.inv(camera_matrix)
		norm_img_point = np.matmul(inv_camera_matrix[0:2,:],pixel_coords_uv)

		# Add distortion
		# Radius
		r_square = norm_img_point[0]**2+norm_img_point[1]**2

		# radial component
		dist_rad_x = norm_img_point[0,:]*(1 + self.distortionVec[0]*r_square + self.distortionVec[1]*r_square**2 + self.distortionVec[2]*r_square**3)
		dist_rad_y = norm_img_point[1,:]*(1 + self.distortionVec[0]*r_square + self.distortionVec[1]*r_square**2 + self.distortionVec[2]*r_square**3)

		# tangential component
		dist_tan_x = 2*self.distortionVec[3]*norm_img_point[0,:]*norm_img_point[1,:] + self.distortionVec[4]*(r_square + 2*norm_img_point[0,:]**2)
		dist_tan_y = 2*self.distortionVec[4]*norm_img_point[0,:]*norm_img_point[1,:] + self.distortionVec[3]*(r_square + 2*norm_img_point[1,:]**2)

		norm_dist_img_points = np.ones((3,pix_v*pix_u))
		norm_dist_img_points[0,:] = dist_rad_x + dist_tan_x
		norm_dist_img_points[1,:] = dist_rad_y + dist_tan_y

		# project distorted points back into image plane
		dist_img_points = np.matmul(camera_matrix[0:2,:],norm_dist_img_points)

		# search corresponding pixels for undistorted image pixels
		undistorted_image = np.zeros((pix_v,pix_u,channels),dtype=np.uint8)
		for idx_point in range(0,pixel_coords_uv.shape[1]):
			# interpolation nearest neighbour
			idx_u = round(dist_img_points[0,idx_point])
			idx_v = round(dist_img_points[1,idx_point])
			if idx_u < 0 or idx_u >= pix_u or idx_v < 0 or idx_v >= pix_v:
				continue
		
			undistorted_image[int(pixel_coords_uv[1,idx_point]),int(pixel_coords_uv[0,idx_point]),:] = image[idx_v,idx_u,:]

		return undistorted_image


if __name__ == "__main__":
	pass
	
	