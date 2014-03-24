function [H] = reconfu2(A,L)
%function [H] = reconfu2(A,L)
% Description: Reconstruction of 2D coordinates with the use of local camera
%         coordinates and the DLT coefficients for one camera.
% Input:  - A  file containing DLT coefficients of the camera
%              [a1 ; a2 ;...]
%         - L  camera coordinates of points
%              [xcam,ycam ;same at time 2]
% Output: - H  global coordinates, residuals,
%              [Xt1,Yt1; same for t2]
% Author: Christoph Reinschmidt, HPL, The University of Calgary
% Date:        September, 1994
% Last change:  November 29, 1996
% Version:     1.1
% Adapted from reconfu.m (3-dimensional DLT)
% Liduin Meershoek, University of Utrecht
% April, 1997


n=size(A,2);
% check whether the numbers of cameras agree for A and L
if size(A,2)~=1 | size(L,2)~=2; disp('there is more then one camera given in A or L')
                   disp('hit any key and then "try" again'); pause; return
end


H(size(L,1),2)=[0];         % initialize H

% ________Building L1, L2:       L1 * G (X,Y) = L2________________________

for k=1:size(L,1)  %number of time points
    L1=[]; L2=[];  % initialize L1,L2
    x=L(k,1); y=L(k,2);
    if ~(isnan(x) | isnan(y))  % do not construct l1,l2 if camx,y=NaN
     L1=[A(1)-x*A(7), A(2)-x*A(8) ; ...
         A(4)-y*A(7), A(5)-y*A(8) ];
     L2=[x-A(3);y-A(6)];
    end

  if (size(L2,1))==2  % check whether data available
   g=L1\L2;
  else
   g=[NaN;NaN];
  end

  H(k,:)=[g'];
end


