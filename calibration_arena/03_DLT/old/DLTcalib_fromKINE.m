clear
clc
close all

% load('DLT_coeff.mat')
load('DLT_coeff_zdown.mat')

[DLT(1).coeff, avgres ] = dltfu( F, L_1, [] );
[DLT(2).coeff, avgres ] = dltfu( F, L_2, [] );
[DLT(3).coeff, avgres ] = dltfu( F, L_3, [] );

% save('DTL_coeff_fromKINE', 'DLT', 'F', 'L','L_1','L_2','L_3')
save('DTL_coeff_zdown_fromKINE', 'DLT', 'F', 'L','L_1','L_2','L_3')