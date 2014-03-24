% File used to test different routines for 3D calibration of caneras.
clear;
close all
data1=[];

[file, pname] = uigetfile('*.mat','Calibration image data file selection');
cd(pname);
load(file);

% % remove first surface
% F = F(25:end,:);
% L = L(25:end,:,:);
% 
% % remove middle surface
% F(25:36,:) = [];
% L(25:36,:,:) = [];
% 
% % remove last surface
% F = F(1:24,:);
% L = L(1:24,:,:);

% world coord (45deg tilt)
Pts = F;

% camera coordinates
Cam1 = L(:,:,1);
Cam2 = L(:,:,2);
Cam3 = L(:,:,3);
% Cam1 = L1;
% Cam2 = L2;
% Cam3 = L3;

% MDLT MOD!
    Meth = 2;

if Meth == 1
    [CamCoef(:,1)]=dltfu(Pts,Cam1);
    [CamCoef(:,2)]=dltfu(Pts,Cam2);
    [CamCoef(:,3)]=dltfu(Pts,Cam3);
elseif Meth == 2
    [dlt1, k1]=mdlt1mod(Pts,Cam1);
    [dlt2, k2]=mdlt1mod(Pts,Cam2);
    [dlt3, k3]=mdlt1mod(Pts,Cam3);
    B1=dlt1(k1,:);
    B2=dlt2(k2,:);
    B3=dlt3(k3,:);
    [CamCoef(:,1)]=B1(:,:)';
    [CamCoef(:,2)]=B2(:,:)';
    [CamCoef(:,3)]=B3(:,:)';
end

% test MDLT coeffs
% modify data
Lmod=[L(:,:,1) L(:,:,2) L(:,:,3)];

[H] = reconfu(CamCoef,Lmod);

figure
plot3(F(:,1),F(:,2),F(:,3),'+')
hold on
plot3(H(:,1),H(:,2),H(:,3),'o')
grid on
axis equal

dev=F-H(:,1:3);
mean_dev=mean(dev(:))
figure
plot3(dev(:,1),dev(:,2),dev(:,3),'.')
grid on
axis equal


% Saving of coefficients to allow use of Peter Maddens Digimat
% fLego=[];
% fCamPoints=[];
% [fn,pn] = uiputfile('*.mat', 'Save CAMERA COEFFICIENT FILE:');
% save([pn, fn], 'CamCoef', 'fLego', 'fCamPoints');
save('MDLT_coeff','CamCoef');

