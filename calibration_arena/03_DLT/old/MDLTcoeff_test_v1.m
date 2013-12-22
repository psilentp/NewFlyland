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
close all

[file, pname] = uigetfile('*.mat','Calibration image data file selection');
cd(pname);
load(file);

[file, pname] = uigetfile('*.mat','Calibration coefficient file selection');
cd(pname);
load(file);

% modify data
Lmod=[L(:,:,1) L(:,:,2) L(:,:,3)];

[H] = reconfu(CamCoef,Lmod)

plot3(F(:,1),F(:,2),F(:,3),'+')
hold on
plot3(H(:,1),H(:,2),H(:,3),'o')
grid on

dev=F-H(:,1:3);
mean_dev=mean(dev(:))
figure
plot3(dev(:,1),dev(:,2),dev(:,3),'.')
grid on
