% reorganize calib data
clear
clc

%% get input variables
N_turns = 6
TPI = 20
TPmm = TPI / 25.4;
D_transl = N_turns / TPmm;

beta = 45 %deg
d_transl(1) =  0;
d_transl(2) = -D_transl * sin(beta);
d_transl(3) =  D_transl * cos(beta);

% origins
Opos1 = [1,0,0]
Opos2 = [1,1,0]
Opos3 = [1,2,0]

%% position 1 (far away, z=0)
load calib_data_pos1

% add initial offset
% offset from corner of checkerboard
X_1(1,:) = X_1(1,:) + dX*Opos1(1);
X_1(2,:) = X_1(2,:) + dX*Opos1(2);
X_1(3,:) = X_1(3,:) + dX*Opos2(3);

F=X_1';

% image coordinates
x(:,:,1)=x_1';
x(:,:,2)=x_2';
x(:,:,3)=x_3';

L=x;

%% position 2 (middle position, z=dz)
load calib_data_pos2

% add initial offset
% ofset from corner of checkerboard
X_1(1,:) = X_1(1,:) + dX*Opos2(1);
X_1(2,:) = X_1(2,:) + dX*Opos2(2);
X_1(3,:) = X_1(3,:) + dX*Opos2(3);

% translational ofset
X_1(1,:) = X_1(1,:) + d_transl(1);
X_1(2,:) = X_1(2,:) + d_transl(2);
X_1(3,:) = X_1(3,:) + d_transl(3);

F(end+1:end+size(X_1,2),:) = X_1';

% image coordinates
clear x
x(:,:,1)=x_1';
x(:,:,2)=x_2';
x(:,:,3)=x_3';

L(end+1:end+size(x,1),:,:)=x;

%% position 3 (front position, z=2*dz)
load calib_data_pos2

% add initial offset
% ofset from corner of checkerboard
X_1(1,:) = X_1(1,:) + dX*Opos3(1);
X_1(2,:) = X_1(2,:) + dX*Opos3(2);
X_1(3,:) = X_1(3,:) + dX*Opos3(3);

% translational ofset
X_1(1,:) = X_1(1,:) + 2*d_transl(1);
X_1(2,:) = X_1(2,:) + 2*d_transl(2);
X_1(3,:) = X_1(3,:) + 2*d_transl(3);

F(end+1:end+size(X_1,2),:) = X_1';

% image coordinates
clear x
x(:,:,1)=x_1';
x(:,:,2)=x_2';
x(:,:,3)=x_3';

L(end+1:end+size(x,1),:,:)=x;

save( 'cali_points', 'F', 'L' );
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

save( 'DLT_coeff', 'DLT','F','L' );