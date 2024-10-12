function [classes] = predict_classes(X, weights, input_layer_size,hidden_layer_size, output_layer_size)
  #AM FOLOSIT ACELASI COD CA LA COST_FUNCTION
  #Facand cost_functions implicand folosirea foward progpagations
  sigmoid = @(x) 1./(1 + e.^(-x));
  derivate_sigmoid = @(x) sigmoid(x).*(1-sigmoid(x));
  totalSize = length(weights);
  sizeO1 = (input_layer_size+1)*hidden_layer_size;
  sizeO2 = output_layer_size*(hidden_layer_size+1);
  O1 = reshape(weights(1:sizeO1),hidden_layer_size,input_layer_size+1);
  O2 = reshape(weights(sizeO1+1:totalSize),output_layer_size,hidden_layer_size+1);
  #calculam foward propagation pentru a obtine H
   [M,N] = size(X);
   z2 = sigmoid(O1 * (X'));
   a2 = [ones(1,M);z2];
   z3 = O2*a2;
   classes = sigmoid(z3)'; 
  endfunction