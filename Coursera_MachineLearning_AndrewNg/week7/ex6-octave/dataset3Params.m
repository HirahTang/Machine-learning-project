function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.03;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
	error = 1;
	for C_opt = [0.01 0.03 0.1 0.3 1 3 10 30]
		for sigma_opt = [0.01 0.03 0.1 0.3 1 3 10 30]
%			fprintf('C value: %f, sigma value: %f', C_opt, sigma_opt);
             	
             model = svmTrain(X, y, C_opt, @(x1, x2) gaussianKernel(x1, x2, sigma_opt));
			 error_temp = mean(double(svmPredict(model, Xval) ~= yval));
			 fprintf('Error @ C = %f sigma = %f: %f\n', C_opt, sigma_opt, error_temp);
			 if (error_temp < error)
				 error = error_temp;
			     C = C_opt;
			     sigma = sigma_opt;
				 fprintf("Better hyperparamters found\n")
			 endif
		end
	end		 



% =========================================================================

end
