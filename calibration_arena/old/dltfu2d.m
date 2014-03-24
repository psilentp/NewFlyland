function [A,avgres] = dltfu2d(F,L,Cut)
% function [A,avgres] = dltfu2d(F,L,Cut)
% Description: Program to calculate 2-dimensional DLT coefficient for one camera
%               Note that at least 4 (valid) calibration points are needed
%               function [A,avgres] = dltfu(F,L,Cut)
% Input:  - F      matrix containing the global coordinates (X,Y)
%                  of the calibration frame
%                  e.g.: [0 20;0 50;0 100;60 20 ...]
%         - L      matrix containing 2d coordinates of calibration 
%                  points seen in camera (same sequence as in F)
%                  e.g.: [1200 1040; 1200 1360; ...]
%         - Cut    points that are not visible in camera;
%                  not being used to calculate DLT coefficient
%                  e.g.: [1 7] -> calibration point 1 and 7 
%                  will be discarded.
%                  This input is optional (default Cut=[]) 
% Output: - A      8 DLT coefficients
%         - avgres average residuals (measure for fit of dlt)
%                  given in units of camera coordinates
%
% Author: Christoph Reinschmidt, HPL, The University of Calgary
% Date:        January, 1994
% Last changes: November 29, 1996
% Version:     1.0
% References:  Woltring and Huiskes (1990) Stereophotogrammetry. In
%               Biomechanics of Human Movement (Edited by Berme and
%               Cappozzo). pp. 108-127.
% Adapted from dltfu.m (3-dimensional DLT)
% Liduin Meershoek, University of Utrecht
% April, 1997
%

if nargin==2; Cut=[]; end;

if size(F) ~= size(L)
disp('# of calibration points entered and seen in camera do not agree'), return
end

m=size(F,1); Lt=L'; C=Lt(:);

for i=1:m
  B(2*i-1,1)  = F(i,1); 
  B(2*i-1,2)  = F(i,2); 
  B(2*i-1,3)  = 1;
  B(2*i-1,7)  =-F(i,1)*L(i,1);
  B(2*i-1,8)  =-F(i,2)*L(i,1);
  B(2*i,4)    = F(i,1);
  B(2*i,5)    = F(i,2);
  B(2*i,6)    = 1;
  B(2*i,7)    =-F(i,1)*L(i,2);
  B(2*i,8)    =-F(i,2)*L(i,2);
end

% Cut the lines out of B and C including the control points to be discarded
Cutlines=[Cut.*2-1, Cut.*2];
B([Cutlines],:)=[];
C([Cutlines],:)=[];

% Solution for the coefficients
A=B\C;
D=B*A;
R=C-D;
res=norm(R); avgres=res/size(R,1)^0.5;





