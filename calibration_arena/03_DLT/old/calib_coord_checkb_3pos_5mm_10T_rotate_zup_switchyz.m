% reorganize calib data
clear
clc
close all

%% get input variables

% origins for 3 positions
% A=0, 1=0 :)

strOpos1=[];
strOpos2=[];
strOpos3=[];

strOpos1 = input('origin at position 1 [default: B3]: ', 's');
if isempty(strOpos1)
    strOpos1 = 'B3'; %deg
end

strOpos2 = input('origin at position 2 [default: B6]: ', 's');
if isempty(strOpos2)
    strOpos2 = 'B6'; %deg
end

strOpos3 = input('origin at position 3 [default: B8]: ', 's');
if isempty(strOpos3)
    strOpos3 = 'B8'; %deg
end

strOpos = [strOpos1 strOpos2 strOpos3];


% while isempty(strOpos1)
%     strOpos1 = input('What is the local origin at position 1 (e.g. b3)? ', 's');
% end
% 
% while isempty(strOpos2)
%     strOpos2 = input('What is the local origin at position 2 (e.g. b3)? ', 's');
% end
% 
% while isempty(strOpos3)
%     strOpos3 = input('What is the local origin at position 3 (e.g. b3)? ', 's');
% end

if strOpos1(1)<97
    Opos1 = [strOpos1(1)-65,strOpos1(2)-49,0];
else
    Opos1 = [strOpos1(1)-97,strOpos1(2)-49,0];
end
    
if strOpos2(1)<97
    Opos2 = [strOpos2(1)-65,strOpos2(2)-49,0];
else
    Opos2 = [strOpos2(1)-97,strOpos2(2)-49,0];
end
    
if strOpos3(1)<97
    Opos3 = [strOpos3(1)-65,strOpos3(2)-49,0];
else
    Opos3 = [strOpos3(1)-97,strOpos3(2)-49,0];
end
    

% cali plate angle
beta = input('Calibration plate angle [default: 45 degrees]: ');
if isempty(beta)
    beta = 45; %deg
end

% plate translation
N_turns = input('Number of turns between positions [default: 10 turns]: ');
if isempty(N_turns)
    N_turns = 10;
end

TPI = input('Number of turns per inch [default: 20 TPI]: ');
if isempty(TPI)
    TPI = 20;
end

TPmm = TPI / 25.4;
D_transl = N_turns / TPmm;

% translation in x', y' and z'
d_transl(1) =  0;
d_transl(2) = -D_transl * sind(beta);
d_transl(3) =  D_transl * cosd(beta);

%% position 1 (far away, z=0)
load calib_data_pos1

% add initial offset
% offset from corner of checkerboard
X_1(1,:) = X_1(1,:) + dX*Opos1(1);
X_1(2,:) = X_1(2,:) + dX*Opos1(2);
X_1(3,:) = X_1(3,:) + dX*Opos1(3);

F=X_1';

% Xmod(1,:) = X_1(2,:);
% Xmod(2,:) = X_1(1,:);
% Xmod(3,:) = X_1(3,:);
% 
% F=Xmod';

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

% clear Xmod
% Xmod(1,:) = X_1(2,:);
% Xmod(2,:) = X_1(1,:);
% Xmod(3,:) = X_1(3,:);
% 
% F(end+1:end+size(X_1,2),:) = Xmod';

% image coordinates
clear x
x(:,:,1)=x_1';
x(:,:,2)=x_2';
x(:,:,3)=x_3';

L(end+1:end+size(x,1),:,:)=x;

%% position 3 (front position, z=2*dz)
load calib_data_pos3

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

% clear Xmod
% Xmod(1,:) = X_1(2,:);
% Xmod(2,:) = X_1(1,:);
% Xmod(3,:) = X_1(3,:);
% 
% F(end+1:end+size(X_1,2),:) = Xmod';

% image coordinates
clear x
x(:,:,1)=x_1';
x(:,:,2)=x_2';
x(:,:,3)=x_3';

L(end+1:end+size(x,1),:,:)=x;

%% rotate coord system

Frot(:,1) = F(:,1);
Frot(:,2) = F(:,2) .* cosd(beta) + F(:,3) .* sind(beta);
Frot(:,3) = F(:,3) .* cosd(beta) - F(:,2) .* sind(beta);

F = Frot;

%% switch yz (z up)

Frot(:,1) = F(:,1);
Frot(:,2) = -F(:,3);
Frot(:,3) = F(:,2);

F = Frot;

%% save data

L_1 = L(:,:,1);
L_2 = L(:,:,2);
L_3 = L(:,:,3);

save( 'cali_points_rotated', 'F', 'L', 'L_1', 'L_2', 'L_3', 'strOpos');
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

% modify data
Lmod=[L(:,:,1) L(:,:,2) L(:,:,3)];
coeff = [DLT(1).coeff DLT(2).coeff DLT(3).coeff];

% Lmod=[L(:,:,2) L(:,:,3)];
% coeff = [DLT(2).coeff DLT(3).coeff];
% 
% Lmod=[L1 L2 L3];
% coeff = CamCoef;

[H] = reconfu(coeff,Lmod);

figure
plot3(F(:,1),F(:,2),F(:,3),'+')
hold on
plot3(H(:,1),H(:,2),H(:,3),'o')
axis equal