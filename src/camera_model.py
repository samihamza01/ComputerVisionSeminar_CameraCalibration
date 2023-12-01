"""@package CameraModel
Author: Sami Hamza

Python implementation of a camera model.
"""

import numpy as np
import numpy.typing as npt
import typing as tp

class CameraModel():
	"""Camera model according to Zhang.
	"""

	def __init__(self, cameraMat: npt.ArrayLike | None, distortionVec: npt.ArrayLike | None, ) -> None:
		pass

	def world_2_image(self, objectPoints: tp.List[tp.List[npt.ArrayLike]]):
		pass

	def _rot_around_axis(self):
		pass

	def calibrate(self, objectPoints: tp.List[tp.List[npt.ArrayLike]], imagePoints: tp.List[tp.List[npt.ArrayLike]]):
		pass
	
	def _world_2_image(self, parameterVec: npt.ArrayLike, objectPoints: tp.List[tp.List[npt.ArrayLike]]):
		pass

	def _residual_function(self, parameterVec: npt.ArrayLike, objectPoints: tp.List[tp.List[npt.ArrayLike]], imagePoints: tp.List[tp.List[npt.ArrayLike]]):
		# determine size of residual vector and create it
		length = 0
		for a in objectPoints:
			length += len(a)

		residualVec = np.zeros((length,))

		# calculate residual vector
		idxResidual = 0
		for idxView, elementView in enumerate(objectPoints):
			for idxPoint, elementPoint in enumerate(elementView):
				estimatedPoint = self.world_2_image(parameterVec,objectPoints[idxView][idxPoint])
				residualVec[idxResidual] = np.linalg.norm(imagePoints[idxView][idxPoint] - estimatedPoint)
				idxResidual += 1

		return residualVec

	def _calibration_plot_reprojection_error(self):
		pass

	def save_cam_to_json():
		pass

	def read_cam_from_json():
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
	worldPoint = np.array([1,1,1])
	rotVec = np.array([0,0,0])
	transVec = np.array([0,0,0])
	distCoeffVec = np.zeros((9,))
	camMat = np.array([[0.002*2000,0,0],[0,0.002*2000,0]])

	print(world_2_image( worldPoint,rotVec,transVec,camMat,distCoeffVec))

	