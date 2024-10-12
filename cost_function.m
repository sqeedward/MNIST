function [grad, J] = cost_function(params, X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size)
  sigmoid = @(x) 1./(1 + e.^(-x));
  derivate_sigmoid = @(x) sigmoid(x).*(1-sigmoid(x));
  totalSize = length(params);
  #impartim Theta-urile in matrici pentru a fi mai usor de folosit
  sizeO1 = (input_layer_size+1)*hidden_layer_size;
  sizeO2 = output_layer_size*(hidden_layer_size+1);
  O1 = reshape(params(1:sizeO1),hidden_layer_size,input_layer_size+1);
  O2 = reshape(params(sizeO1+1:totalSize),output_layer_size,hidden_layer_size+1);
  #calculam foward propagation pentru a obtine H
    [M,N] = size(X);
   z2 = sigmoid(O1 * (X'));
   a2 = [ones(1,M);z2];
   z3 = O2*a2;
   a3 = sigmoid(z3);
   H0 = a3;
   bigY = zeros(10,M);
   for i = [1:M]
     bigY(y(i),i) = 1;
   endfor
   grad = 0;
   #calcuam costul cu J ca na
   cost = sum(-bigY.*log(H0)-(1-bigY).*log(1-H0))';
   J = sum(cost)/M+(lambda/(2*M))*(sum(sum(O1.^2))+sum(sum(O2.^2)));
   #aici calculam gradientul
   D1 = zeros(hidden_layer_size,input_layer_size+1);
   D2 = zeros(output_layer_size,hidden_layer_size+1);
   
   Err1 = a3-bigY;
   D2 = D2 + Err1*a2';
   Err2 = (O2)'*Err1.*derivate_sigmoid(a2);
   Err2 = Err2(2:hidden_layer_size+1,:);
   D1 = D1+ Err2*X;
   #deltile, adica noii neuroni (imi mor mie neuronii ca ecuatiile din ex nu sunt corecteee)
   deltaJ1 = D1/M;
   deltaJ2 = D2/M;
   diffO1 = (lambda/M)*O1;
   diffO2 = (lambda/M)*O2;
   diffO1(1,1) = 0;
   diffO2(1,1) = 0;
   deltaJ1 = deltaJ1 + diffO1;
   deltaJ2 = deltaJ2 + diffO2;
   deltaJ1 = reshape(deltaJ1,sizeO1,1);
   deltaJ2 = reshape(deltaJ2,sizeO2,1);
   grad = [deltaJ1;deltaJ2];
  endfunction