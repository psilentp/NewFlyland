%function [H] = reconfu(A,L)
% Description:	Reconstruction of 3D coordinates with the use local (camera
%		coordinates and the DLT coefficients for the n cameras).
% Input:	- A  file containing DLT coefficients of the n cameras
%		     [a1cam1,a1cam2...;a2cam1...]
%		- L  camera coordinates of points
%		     [xcam1,ycam1,xcam2,ycam2...;same at time 2]
% Output:	- H  global coordinates, residuals, cameras used
%		     [Xt1,Yt1,Zt1,residt1,cams_used@t1...; same for t2]
% Author:	Christoph Reinschmidt, HPL, The University of Calgary
% Date:		September, 1994
% Last change:  November 29, 1996
% Version:	1.1
clc
clear

[file, pname] = uigetfile('*.mat','Calibration image data file selection');
cd(pname);
load(file);

[file, pname] = uigetfile('*.mat','Calibration coefficient file selection');
cd(pname);
load(file);

% modify data
Lmod=[L(:,:,1) L(:,:,2) L(:,:,3)];
coeff = [DLT(1).coeff DLT(2).coeff DLT(3).coeff];

% Lmod=[L1 L2 L3];
% coeff = CamCoef;

[H] = reconfu(coeff,Lmod)

figure
plot3(F(:,1),F(:,2),F(:,3),'.')
hold on
plot3(H(:,1),H(:,2),H(:,3),'or')
