"""@package:	LevenbergMarquardt
Author:	Sami Hamza

Python implementation of the Levenberg Marquardt Algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

# typing
import numpy.typing as npt
import typing as tp

class LevenbergMarquardtOptimizer():
	"""Levenberg Marquardt Algorithm according to madsen alg. 3.16.
	"""

	# attributes
	maxIterations: int
	gradientThr: float
	parameterStepThr: float

	lamb: float

	def __init__(self, maxIterations=200, gradientThr=1e-8, parameterStepThr=1e-8) -> None:
		"""Constrictor.

		Args:
			maxIterations (tp.Callable): 			Value specifying the maximum number of Iterations (default = 400).
			gradientThr (npt.ArrayLike): 			Convergence criteria threshold for the gradient specified by a small value (default = 1e-8).
			parameterStepThr (npt.ArrayLike): 		Convergence criteria threshold for the parameter step specified by a small value (default = 1e-8).

		Returns:
			None
		"""

		self.maxIterations = maxIterations
		self.gradientThr = gradientThr
		self.parameterStepThr = parameterStepThr
		return
	
	def _cal_jacobian(self, residualFunction: tp.Callable, parameterVector: npt.ArrayLike, inputVector: npt.ArrayLike, referenceVector: npt.ArrayLike) -> npt.ArrayLike:
		"""Helper Function to calculate the jacobian of the given residual function.

		Args:
			residualFunction (tp.Callable): 			Residualfunction. Must be a function of the parametervector (paramVector),
														the input (inputVector) and the reference data (referenceVector).
			parameterVector (npt.ArrayLike): 			Vector containing the function parameters (shape: (n,)).
			inputVector (npt.ArrayLike): 				Vector of input to corresponding reference.
			referenceVector (npt.ArrayLike): 			Vector containing all reference/ground truth data.

		Returns:
			npt.ArrayLike: Jacobian matrix.
		"""
		

		m = referenceVector.shape[0]
		n = parameterVector.shape[0]

		h = 1e-8
		eye = np.eye(n,dtype=np.float64) * h

		jacobianMat = np.zeros(shape=(m, n),dtype=np.float64)

		for i, _ in enumerate(parameterVector):
			
			r_plus = residualFunction(parameterVector + eye[i], inputVector, referenceVector)
			r_minus = residualFunction(parameterVector - eye[i], inputVector, referenceVector)

			central_difference = (r_plus - r_minus) / (2*h)

			jacobianMat[:, i] = central_difference
		return jacobianMat


	def optimize(self, residualFunction: tp.Callable, parameterVector: npt.ArrayLike, inputVector: npt.ArrayLike, referenceVector: npt.ArrayLike) -> tp.Tuple[npt.ArrayLike,int,float,float,npt.ArrayLike,npt.ArrayLike,int]:
		"""Function to optimze the specified problem

		Args:
			residualFunction (tp.Callable): 			Residualfunction. Must be a function of the parametervector (paramVector),
														the input (inputVector) and the reference data (referenceVector).
			parameterVector (npt.ArrayLike): 			Vector containing the function parameters.
			inputVector (npt.ArrayLike): 				Vector of input to corresponding reference.
			referenceVector (npt.ArrayLike): 			Vector containing all reference/ground truth data.

		Returns:
			tp.Tuple[npt.ArrayLike,float,float,int]: 	A tupel of the optimal parameters, the convergence/stopping reason 
														{0: in gradient, 1: in paramerters, 2: max iterations},
														the final error of the least squares problem,
														the mean of the residuals, the standart deviation of the parameters,
														the relative error of the parameters
														and the iteration where convergence was reached.
		"""

		iteration = 0
		v = 2
		tau = 1e-13
		
		# calculate jacobian
		jacobianMat = self._cal_jacobian(residualFunction, parameterVector, inputVector, referenceVector)
		# calculate the inforamtion matrix (approximation of problems hessian)
		infMat = np.matmul(jacobianMat.T,jacobianMat)
		# calculate gradient
		residualVector = residualFunction(parameterVector, inputVector, referenceVector)
		grad = np.matmul(jacobianMat.T,residualVector)

		# determine the stating value of lambda
		lamb = tau * np.max(np.diag(infMat))

		# convergence reason {0: in gradient, 1: in paramerters, 2: max iterations}
		conv = -1

		# convergence condition
		found = np.max(np.abs(grad)) <= self.gradientThr

		SquareError = []
		while (not found) and (iteration <= self.maxIterations) :
			# History logging
			SquareError.append(np.matmul(residualVector.T,residualVector))
			# solve normal equation for parameter update
			parameterStep = np.linalg.solve(infMat + lamb*np.diag(np.diag(infMat)), -grad)

			# check for convergence in parameterStep
			if np.linalg.norm(parameterStep) <= self.parameterStepThr*(np.linalg.norm(parameterVector) + self.parameterStepThr):
				conv = 1
				found = True
			else:
				parameterVectorNew = parameterVector + parameterStep
				# calculate gain ratio
				residualVectorNew = residualFunction(parameterVectorNew, inputVector, referenceVector)
				change = np.matmul(residualVector.T,residualVector) - np.matmul(residualVectorNew.T,residualVectorNew)
				estimatedChange = np.matmul(parameterStep.T, lamb*np.matmul(np.diag(np.diag(infMat)),parameterStep) - grad)
				gain_ratio = change/estimatedChange

				if gain_ratio > 0:
					# step accepted
					# set new parametervector
					parameterVector = parameterVector + parameterStep
					residualVector = residualVectorNew
					# calc new information matrix and gradient
					jacobianMat = self._cal_jacobian(residualFunction, parameterVector, inputVector, referenceVector)
					infMat = np.matmul(jacobianMat.T,jacobianMat)
					grad = np.matmul(jacobianMat.T,residualVector)
					# check convergence in gradient
					found = np.max(np.abs(grad)) <= self.gradientThr

					# update lambda
					lamb = lamb*np.max([1/3, 1 - (2*gain_ratio-1)**3])
					v = 2
				else:
					# step not accepted
					# update lambda
					lamb = lamb * v
					v = 2*v
			iteration += 1 
		
		if (conv != 1) and iteration >= self.maxIterations:
			conv = 2
		elif conv != 1:
			conv = 0
		

		# error (sensitivity analysis)
		covar_parameters = np.linalg.inv(infMat)
		stddev_parameters = np.sqrt(np.diag(covar_parameters))
		error_parameters = None#stddev_parameters/parameterVector

		# final error of the least square problem
		error = np.matmul(residualVector.T,residualVector)
		# mean of the residuals
		meanResidual = np.sum(residualVector)/residualVector.shape[0]
		return (parameterVector, conv, error, meanResidual, stddev_parameters, error_parameters, iteration-1, SquareError)
	

if __name__ == "__main__":
	print("Example of nonlinear least squares optimization with Levenberg Marquardt Algorithm")
	print("Exponential fitting: a*exp(b*x)")
	
	# define the function to fit and the residual function
	def exampleFunc(parameterVec, inputVec):
		return parameterVec[0]*np.exp( parameterVec[1]*inputVec)
	def residualFunc(parameterVec, inputVec, referenceVec):
		return referenceVec - exampleFunc(parameterVec, inputVec)

	# simulate reference data (normally the data to fit the function to)
	inp = np.linspace(0,10,100)
	params = np.array([2.,-1.8])
	y = exampleFunc(params, inp)
	#print(f"Reference data:\n{y}")

	# initial parameter guess
	initParams = np.array([1.,0.])
	print(f"Initial params: a={initParams[0]}, b={initParams[1]}")
	levMarq = LevenbergMarquardtOptimizer(30,1e-8,1e-8)
	paramsOpt, conv, error, meanResidual, stddev_parameters, error_parameters, iteration, squareErrorHist = levMarq.optimize(residualFunc,initParams,inp, y)
	if conv == 0:
		print(f"Convergence in gradient after {iteration} iterations.")
	elif conv == 1:
		print(f"Convergence in parameters after {iteration} iterations.")
	else:
		print(f"No convergence reached after max iterations [{iteration}]!")
	print(f"Optimized params: {paramsOpt}")
	print(f"Final squared error: {error}")
	print(f"Mean residuals: {meanResidual}")
	print("Sensitivity:")
	print(f"Standard deviation params: {stddev_parameters}")
	print(f"Standard error params: {error_parameters}")

	# plot history
	ax = plt.subplot(111)
	ax.plot(np.linspace(0,iteration,iteration+1),squareErrorHist)
	ax.set_title("Convergence Plot")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Square Error")
	ax.set_xlim(0,iteration)
	ax.set_ylim(bottom=0)
	plt.show()


	