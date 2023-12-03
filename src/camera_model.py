"""@package CameraModel
Author: Sami Hamza

Python implementation of a camera model.
"""

import numpy as np
import json
import levenberg_marquardt as lm
import numpy.typing as npt
import typing as tp

class CameraModel():
	"""Camera model according to Zhang.
	"""
	cameraMat: npt.ArrayLike
	distortionVec: npt.ArrayLike
	imageSize: npt.ArrayLike
	
	def __init__(self, cameraMat: tp.Optional[npt.ArrayLike] = None, distortionVec: tp.Optional[npt.ArrayLike] = None, imageSize: tp.Optional[npt.ArrayLike] = None) -> None:
		"""_summary_

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
			self.distortionVec = np.zeros((17,1),dtype=float)
		else:
			self.distortionVec = distortionVec

		if imageSize is None:
			self.imageSize = np.zeros((2,1),dtype=int)
		else:
			self.imageSize = imageSize

		return

	def world_2_image(self, objectPoints: tp.List[tp.List[npt.ArrayLike]]) -> npt.ArrayLike:
		# TODO
		pass

	def _rot_around_axis(vector: npt.ArrayLike,axis: npt.ArrayLike, angle: float) -> npt.ArrayLike:
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

		rotMat = np.array(	[cAng + axis[0]**2*s_CAng, axis[0]*axis[1]*s_CAng - axis[2]*sAng, axis[0]*axis[2]*s_CAng + axis[1]*sAng],
							[axis[0]*axis[1]*s_CAng + axis[2]*sAng, cAng + axis[1]**2*s_CAng, axis[1]*axis[2]*s_CAng - axis[0]*sAng],
							[axis[0]*axis[2]*s_CAng - axis[1]*sAng, axis[1]*axis[2]*s_CAng + axis[0]*sAng, cAng + axis[2]**2*s_CAng])

		v_rot = np.matmul(rotMat, vector)
		return v_rot

	def calibrate(self, objectPoints: tp.List[tp.List[npt.ArrayLike]], imagePoints: tp.List[tp.List[npt.ArrayLike]], imageSize: npt.ArrayLike):
		levMar = lm.LevenbergMarquardtOptimizer()
		# TODO: add homographie part for initial values-----
		# get inial parameters via homographie
		parameterVec = np.zeros(15,1)
		# --------------------------------------------------
		optimalParams,_,_,_,_,_ = levMar.optimize(self._residual_function,parameterVec,objectPoints,imagePoints)
		self.cameraMat = np.array([	[optimalParams[0],0,optimalParams[2]],
									[0,optimalParams[1],optimalParams[3]],
									[0,0,1]])
		self.distortionVec = optimalParams[4:8]
		self.imageSize = imageSize.copy()
		return
	
	def _world_2_image(self, parameterVec: npt.ArrayLike, objectPoint: npt.ArrayLike) -> npt.ArrayLike:
		# Transform from World to Cameracoordinates
		if np.linalg.norm(parameterVec[9:11]) != 0:
			camPoint = rot_around_axis(objectPoint,parameterVec[9:11]/np.linalg.norm(parameterVec[9:11]),np.linalg.norm(parameterVec[9:11])) + parameterVec[12:14]
		else:
			camPoint = objectPoint + parameterVec[12:14]

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

		# # thin prism component
		# distPrismX = parameterVec[9]*r_square + parameterVec[10]*r_square**2
		# distPrismY = parameterVec[11]*r_square + parameterVec[12]*r_square**2

		normDistImgPoint = np.array([	distRadX + distTanX,
										distRadY + distTanY,
										1])

		# Project in image plane
		camMat = np.array([	[parameterVec[0],0,parameterVec[2]],
							[0,parameterVec[1],parameterVec[3]],
							[0,0,1]])
		imgPoint = np.matmul(camMat,normDistImgPoint)

		return imgPoint

	def _residual_function(self, parameterVec: npt.ArrayLike, objectPoints: tp.List[tp.List[npt.ArrayLike]], imagePoints: tp.List[tp.List[npt.ArrayLike]]) -> npt.ArrayLike:
		"""Function calculating the residual vector for optimization with Levenberg Marquardt.

		Args:
			parameterVec (npt.ArrayLike): Parameter vector containing (fx,fy,cx,cy,k1,k2,k3,p1,p2,r1,r2,r3,t1,t2,t3).
			objectPoints (tp.List[tp.List[npt.ArrayLike]]): Object point list, where the outer list contains the view sets and the inner list contains the points per view.
			imagePoints (tp.List[tp.List[npt.ArrayLike]]): Image point list, where the outer list contains the view sets and the inner list contains the points per view

		Returns:
			npt.ArrayLike: Residual Vector.
		"""
		# determine size of residual vector and create it
		length = 0
		for a in objectPoints:
			length += len(a)

		residualVec = np.zeros((length,))

		# parameter vector
		parameterViewVector = np.zeros(15,)
		parameterViewVector[0:8] = parameterVec[0:8]

		# calculate residual vector
		idxResidual = 0
		paramOffset =  len(objectPoints)*3 + 8
		for idxView, elementView in enumerate(objectPoints):
			# get current view params
			parameterViewVector[9:11] =  parameterVec[8+3*idxView+1:11+idxView*3]
			parameterViewVector[12:14] =  parameterVec[paramOffset+3*+idxView+1:paramOffset+3+3*idxView]
			for idxPoint, elementPoint in enumerate(elementView):
				estimatedPoint = self.world_2_image(parameterViewVector,objectPoints[idxView][idxPoint])
				residualVec[idxResidual] = np.linalg.norm(imagePoints[idxView][idxPoint] - estimatedPoint)
				idxResidual += 1

		return residualVec

	def _calibration_plot_reprojection_error(self):
		# TODO
		pass

	def save_cam_to_json(self, path: str):
		camData = {"cameraMat": self.cameraMat.tolist(), "distortionVec": self.distortionVec.tolist()}
		with open(path,'w') as f:
			json.dump(camData,f)
		return

	def load_cam_from_json(self, path: str):
		with open(path,'r') as f:
			camData = json.load(f)
		self.cameraMat = np.array(camData["cameraMat"])
		self.distortionVec = np.array(camData["distortionVec"])
		return

	def undistortImage():
		pass


def rot_around_axis(vector: npt.NDArray[tp.Any],axis: npt.NDArray[tp.Any], angle: float) -> npt.NDArray[tp.Any]:
	"""
	@brief Function to rotate a vector around the normed axis n by the specified angle.

	@param 3D vector Vector to rotate.
	@param 3D normed axis of rotation.
	@param Angle to rotate in rad. The sign is defined by the right hand roule.

	@return Rotated vector.
    """
	# perform a rotation according to rodrigues
	# v' = v*cos(phi)+(axis x v)sin(phi)+axis * (axis * v)*(1-cos(phi))
	# with phi = norm(axis)
	# as matrix (Rotationmatrix) => faster
	# reuse the cos and sin
	cAng = np.cos(angle)
	sAng = np.sin(angle)
	# reuse the value of 1-cos (shifted cos)
	s_CAng = 1 - cAng

	rotMat = np.array(	[cAng + axis[0]**2*s_CAng, axis[0]*axis[1]*s_CAng - axis[2]*sAng, axis[0]*axis[2]*s_CAng + axis[1]*sAng],
						[axis[0]*axis[1]*s_CAng + axis[2]*sAng, cAng + axis[1]**2*s_CAng, axis[1]*axis[2]*s_CAng - axis[0]*sAng],
						[axis[0]*axis[2]*s_CAng - axis[1]*sAng, axis[1]*axis[2]*s_CAng + axis[0]*sAng, cAng + axis[2]**2*s_CAng])

	v_rot = np.matmul(rotMat, vector)
	return v_rot

def world_2_image(worldPoint: npt.NDArray[tp.Any],rotVec: npt.NDArray[tp.Any], transVec: npt.NDArray[tp.Any], camMat: npt.NDArray[tp.Any], distCoeffVec: npt.NDArray[tp.Any]) -> npt.NDArray[tp.Any]:
	"""
	@brief Function to project a world point into the image plane.

	@param worldPoint 	Point to project [x_w, y_w, z_w].
	@param rotVec 		Vector specifing the rotation axis and angle (norm(rotVec)).
	@param transVec 	Translation from world to cameracoordinatesystem.
	@param camMat 		Cameramatrix containing fx, fy, cx, cy.
	@param distCoeffVec	Vector containing the distortioncoefficients [k1, k2, k3, p1, p2, s1, s2, s3, s4]

	@return The corresponding image point to a world point.
    """
	
	# Transform from World to Cameracoordinates
	if np.linalg.norm(rotVec) != 0:
		camPoint = rot_around_axis(worldPoint,rotVec/np.linalg.norm(rotVec),np.linalg.norm(rotVec)) + transVec
	else:
		camPoint = worldPoint + transVec

	# Project in normalised image plane
	normImgPoint = np.array(	[camPoint[0]/camPoint[2],
								camPoint[1]/camPoint[2],
								1])

	# Add distortion
	# Radius
	r_square = normImgPoint[0]**2+normImgPoint[1]**2

	# radial component
	distRadX = normImgPoint[0]*(1 + distCoeffVec[0]*r_square + distCoeffVec[1]*r_square**2 + distCoeffVec[2]*r_square**3)
	distRadY = normImgPoint[1]*(1 + distCoeffVec[0]*r_square + distCoeffVec[1]*r_square**2 + distCoeffVec[2]*r_square**3)

	# tangential component
	distTanX = 2*distCoeffVec[3]*normImgPoint[0]*normImgPoint[1] + distCoeffVec[4]*(r_square + 2*normImgPoint[0]**2)
	distTanY = 2*distCoeffVec[4]*normImgPoint[0]*normImgPoint[1] + distCoeffVec[3]*(r_square + 2*normImgPoint[1]**2)

	# thin prism component
	distPrismX = distCoeffVec[5]*r_square + distCoeffVec[6]*r_square**2
	distPrismY = distCoeffVec[7]*r_square + distCoeffVec[8]*r_square**2

	normDistImgPoint = np.array([	distRadX + distTanX + distPrismX,
									distRadY + distTanY + distPrismY,
									1])

	# Project in image plane
	imgPoint = np.matmul(camMat,normDistImgPoint)

	return imgPoint

if __name__ == "__main__":
	cam = CameraModel()
	cam.save_cam_to_json("test.json")
	cam.load_cam_from_json("test.json")
	# worldPoint = np.array([1,1,1])
	# rotVec = np.array([0,0,0])
	# transVec = np.array([0,0,0])
	# distCoeffVec = np.zeros((9,))
	# camMat = np.array([[0.002*2000,0,0],[0,0.002*2000,0]])

	# print(world_2_image( worldPoint,rotVec,transVec,camMat,distCoeffVec))

	