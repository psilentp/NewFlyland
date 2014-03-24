% Get DLT calibration

clear all

save_file = 'DLT_coeff.mat';

% % Get calibration test points
% load all_calib_pts.mat

% Get calibration test points from checkerboard
load calib_data.mat

% convert data

F=X_1'; % world coord

% image coord
L1=x_1';
L2=x_2';
L3=x_3';

L(:,:,1)=L1;
L(:,:,2)=L2;
L(:,:,3)=L3;

% Get DLT coeff 1
[ DLT_1, avgres ] = dltfu( F, L(:,:,1), [] );

% Get DLT coeff 1
[ DLT_2, avgres ] = dltfu( F, L(:,:,2), [] );

% Get DLT coeff 1
[ DLT_3, avgres ] = dltfu( F, L(:,:,3), [] );

% Save calibration 
DLT(1).coeff = DLT_1;

DLT(1).cam = 'cam1';

DLT(2).coeff = DLT_2;

DLT(2).cam = 'cam2';

DLT(3).coeff = DLT_3;

DLT(3).cam = 'cam3';

save( save_file, 'DLT' );