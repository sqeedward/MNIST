function [J,cost] = calculate_cost(X,Y,O1,O2,alpha)
  [M,N] = size(X);
  sigmoid = @(x) 1./(1 + e.^(-x));
   z = sigmoid(O1 * (X'));
   z = [ones(1,M);z];
   H0 = sigmoid(O2*z);
   bigY = zeros(10,M);
   for i = [1:M]
     bigY(Y(i),i) = 1;
   endfor
   cost = sum(-bigY.*log(H0)-(1-bigY).*log(1-H0))';
   J = sum(cost)/M+(alpha/(2*M))*(sum(sum(O1.^2))+sum(sum(O2.^2)));
  endfunction