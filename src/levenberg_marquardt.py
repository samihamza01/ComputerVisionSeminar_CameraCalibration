"""!@package:	LevenbergMarquardt
Author:	Sami Hamza

Python implementation of the Levenberg Marquardt Algorithm.
"""

import numpy as np
import numpy.typing as npt
import typing as tp

class LevenbergMarquardOptimizer():
	"""!@class:	LevenbergMarquardt 
		@brief:	Algorithm according to madsen alg. 3.16.
	"""


	def __init__(self, referenceVector, mappingFunction, paramVector, maxIterations, gradientThr, paramStepThr) -> None:
		"""!@brief:	ctor.

			@param referenceVector:	Vector containing all reference/ground truth data.
			@param mappingFunction:	Function that needs to be fitted.
			@param paramVector:		Vector containing the function parameters.
			@param maxIterations: 	Value specifying the maximum number of Iterations.
			@param gradientThr:		Convergence criteria threshold for the gradient specified by a small value.
			@param paramStepThr:		Convergence criteria threshold for the parameter step specified by a small value.
		"""
		pass
	
	def cal_jacobian(self) -> npt.NDArray:
		"""!@brief:	Helper method to calculate the jacobian of the mapping function. The jacobian can be approximated analytically
					or numerically.

			@param

			@return:
		"""
		pass

	def cal_gain_ratio(self) -> float:
		"""!@brief:	Helper method to calculate the parameterstep evaluation metric.

			@param

			@return:
		"""
		pass

	def update_lambda(self) -> float:
		"""!@brief:	Helper method to update the marquardt parameter lambda. [see garvin + madsen]

			@param

			@return:
		"""
		pass

	def solve_LM_for_paramUpdate(self) -> npt.NDArray(tp.Any):
		"""!@brief:	Helper method to solve LM-equation for parameter update vector. This can be done by solving the normal 
					equations or by performing the orthogonal transformation. Second will be more precise, but also more expensive.

			@param

			@return:
		"""
		pass

	def begin(self) -> npt.NDArray(tp.Any):
		"""!@brief:	Performs the optimization.

			@return:	The minimized scalar value of the non linear least square problems objective function.
		"""
		pass
	

if __name__ == "__main__":
	pass

	