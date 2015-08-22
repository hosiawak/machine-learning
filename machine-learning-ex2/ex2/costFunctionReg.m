function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sumDelta = 0;

for iter = 1:m
  gx = theta' * X(iter,:)';
  hx = sigmoid(gx);
  delta = -y(iter) * log(hx) - (1 - y(iter)) * log(1 - hx)
  sumDelta = sumDelta + delta;
  J = sumDelta / m;

  for gradIter = 1:size(theta, 1)
    deltaGradient = (hx - y(iter)) * X(iter, gradIter);
    grad(gradIter) = grad(gradIter) + deltaGradient;
  end
end

grad = grad / m;

				% Add regularization

theta1 = theta;
theta1(1) = 0;

reg = (lambda / (2 * m)) * sum(theta1 .^ 2);

J = J + reg;

grad = grad + lambda * theta1 / m;
% =============================================================

end
