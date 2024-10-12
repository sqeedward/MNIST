function [x_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  #Amestecam cum zic baietii
  #percent = percent/100;
  U = length(y);
  #de modificat daca vor floor baietii
  train_number = ceil(percent*U)
  x_train = X([1:train_number],:);
  y_train = y([1:train_number],:);
  X_test = X([train_number+1:U],:);
  y_test = y([train_number+1:U],:);
  
  endfunction