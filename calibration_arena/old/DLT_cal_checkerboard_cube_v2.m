% reorganize calib data

clear
clc

save_file = 'DLT_coeff.mat';

%% plate A, cam I & cam III
load calib_data_A

% add initial offset (from corner of cube)
X_1(1,:) = X_1(1,:) + dX;
X_1(2,:) = X_1(2,:) + dX;

F=X_1';

% image coord for cam I & cam III
x(:,:,1)=x_1';
x(:,:,2)=NaN;
x(:,:,3)=x_3';

L=x;

%% plate B, cam I & cam II
load calib_data_B

% add initial offset (from corner of cube)
X_1(1,:) = X_1(1,:) + dX;
X_1(2,:) = X_1(2,:) + dX;

% change coord: x->z,y->x,z->y
X(1,:) = X_1(3,:);
X(2,:) = X_1(1,:);
X(3,:) = X_1(2,:);

F(end+1:end+size(X,2),:) = X';

% image coord for cam I & cam II
clear x
x(:,:,1)=x_1';
x(:,:,2)=x_2';
x(:,:,3)=NaN;

L(end+1:end+size(x,1),:,:)=x;

%% plate C, cam II & cam III
load calib_data_C

% add initial offset (from corner of cube)
X_2(1,:) = X_2(1,:) + dX;
X_2(2,:) = X_2(2,:) + dX;

% change coord: x->y,y->z,z->x
clear X
X(1,:) = X_2(2,:);
X(2,:) = X_2(3,:);
X(3,:) = X_2(1,:);

F(end+1:end+size(X,2),:) = X';

% image coord for cam II & cam III
clear x
x(:,:,2)=x_2';
x(:,:,3)=x_3';
x(:,:,1)=NaN;

L(end+1:end+size(x,1),:,:)=x;

%% Get DLT calibration

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