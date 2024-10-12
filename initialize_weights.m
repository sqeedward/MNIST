function [matrix] = initialize_weights(L_prev, L_next)
 epsilon0 = sqrt(6)/sqrt(L_prev+L_next);
 matrix = epsilon0*(2*rand(L_next,L_prev+1)-1);
endfunction