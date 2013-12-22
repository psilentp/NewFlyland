function b=mdlt1(pk,sk)

%Author: 	Tomislav Pribanic, University of Zagreb, Croatia
%e-mail:		Tomislav.Pribanic@zesoi.fer.hr
%				Any comments and suggestions would be more than welcome.
%Date:		September 1999
%Version: 	1.0

%Function uses MDLT method adding non-linear constraint:
%(L1*L5+L2*L6+L3*L7)*(L9^2+L10^2+L11^2)=(L1*L9+L2*L10+L3*L11)*(L5*L9+L6*L10+L7*L11) (1)
%(assuring orthogonality of transformation matrix and eliminating redundant parametar) to the
%linear system of DLT represented by basic DLT equations:
%							u=(L1*X+L2*Y+L3*Z+L4)/(L9*X+L10*Y+L11*Z+1);	(2)
%							v=(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1);	(3).
%(u,v)	image coordinates in digitizer units
%L1...L11 	DLT parameters
%(X,Y,Z)	object space coordinates
%Once the non-linear equation (1) was solved for the L1 parameter, it was substituted
% for L1 in the equation (2) and now only 10 DLT parameters appear.

%The obtained non-linear system was solved with the following algorithm (Newton's method):
%equations u=f(L2-L11) (2) and v=g(L2-L11) (3) were expressed using truncated Taylor
%series expansion (up to the first derivative) resulting again with 
%the set of linearized equations (for particular point we have):
%	u=fo+pd(fo)/pd(L2)*d(L2)+...+pd(fo)/pd(L11)*d(L11)		(4)
%	v=go+pd(go)/pd(L2)*d(L2)+...+pd(go)/pd(L11)*d(L11)		(5)
%pd- partial derivative
%d-differential
%fo, go, pd(fo)/pd(L2)...pd(fo)/pd(L11)*d(L11) and  pd(go)/pd(L2)...pd(go)/pd(L11) are
%current estimates acquired by previous iteration.
%Initial estimates are provided by conventional 11 DLT parameter method.

%Therefore standard linear least square technique can be applied to calculate d(L2)...d(L11)
%elements.
%Each element is in fact d(Li)=Li(current iteration)-Li(previous iteration, known from before).
%Li's of current iteration can be than substituted for a new estimates in (4) and (5) until
%all elements of d(Li's) are satisfactory small.

%REFERENCES:

%1. The paper explains linear and non-linear MDLT.
%	 The function reflects only the linear MDLT (no symmetrical or
%	 asymmetrical lens distortion parameters included).

%   Hatze H. HIGH-PRECISION THREE-DIMENSIONAL PHOTOGRAMMETRIC CALIBRATION
%   AND OBJECT SPACE RECONSTRUCTION USING A MODIFIED DLT-APPROACH.
%   J. Biomechanics, 1988, 21, 533-538

%2. The paper shows the particular mathematical linearization technique for 
%	 solving non-linear nature of equations due to adding non-linear constrain.

%	 Miller N. R., Shapiro R., and McLaughlin T. M. A TECHNIQUE FOR OBTAINING
%	 SPATIAL KINEMATIC PARAMETERS OF SEGMENTS OF BIOMECHANICAL SYSTEMS 
%	 FROM CINEMATOGRAPHIC DATA. J. Biomechanics, 1980, 13, 535-547




%Input:		pk-matrix containing global coordinates (X,Y,Z) of the ith point
%				e.g. pk(i,1), pk(i,2), pk(i,3)
%				sk-matrix containing image coordinates (u,v) of the ith point
%    			e.g. sk(i,1), sk(i,2)
%Output:		sets of 11 DLT parameters for all iterations
%				The code is far from being optimal and many improvements are to come.

%[a]*[b]=[c]
m=size(pk,1);	% number of calibration points
c=sk';c=c(:);	% re-grouping image coordinates in one column
ite=10; 			%number of iterations

% Solve 'ortogonality' equation (1) for L1
L1=solve('(L1*L5+L2*L6+L3*L7)*(L9^2+L10^2+L11^2)=(L1*L9+L2*L10+L3*L11)*(L5*L9+L6*L10+L7*L11)','L1');
%initialize basic DLT equations (2) and (3)
u=sym('(L1*X+L2*Y+L3*Z+L4)/(L9*X+L10*Y+L11*Z+1)');
v=sym('(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1)');
%elimenate L1 out of equation (2)using the (1)
jed1=[ char(L1) '=L1'];
jed2=[ char(u) '=u'];
[L1,u]=solve( jed1, jed2,'L1,u');

%Find the first partial derivatives of (4) and (5)
%f(1)=diff(u,'L1');g(1)=diff(v,'L1'); 
%L1 was chosen to be eliminated. In case other parameter (for example L2) is chosen
%the above line should become active and the appropriate one passive instead.
f(1)=diff(u,'L2');g(1)=diff(v,'L2');
f(2)=diff(u,'L3');g(2)=diff(v,'L3');
f(3)=diff(u,'L4');g(3)=diff(v,'L4');
f(4)=diff(u,'L5');g(4)=diff(v,'L5');
f(5)=diff(u,'L6');g(5)=diff(v,'L6');
f(6)=diff(u,'L7');g(6)=diff(v,'L7');
f(7)=diff(u,'L8');g(7)=diff(v,'L8');
f(8)=diff(u,'L9');g(8)=diff(v,'L9');
f(9)=diff(u,'L10');g(9)=diff(v,'L10');
f(10)=diff(u,'L11');g(10)=diff(v,'L11');

%Find the inital estimates using conventional DLT method
for i=1:m
   a(2*i-1,1)=pk(i,1);
   a(2*i-1,2)=pk(i,2);
   a(2*i-1,3)=pk(i,3);
   a(2*i-1,4)=1;
   a(2*i-1,9)=-pk(i,1)*sk(i,1);
   a(2*i-1,10)=-pk(i,2)*sk(i,1);
   a(2*i-1,11)=-pk(i,3)*sk(i,1);
   a(2*i,5)=pk(i,1);
   a(2*i,6)=pk(i,2);
   a(2*i,7)=pk(i,3);
   a(2*i,8)=1;
   a(2*i,9)=-pk(i,1)*sk(i,2);
   a(2*i,10)=-pk(i,2)*sk(i,2);
   a(2*i,11)=-pk(i,3)*sk(i,2);
end
%Conventional DLT parameters
b=a\c;
%Take the intial estimates for parameters
%L1=b(1); L1 is excluded.
L2=b(2);L3=b(3);L4=b(4);L5=b(5);L6=b(6);
L7=b(7);L8=b(8);L9=b(9);L10=b(10);L11=b(11);
clear a b c

%Perform the linear least square technique on the system of equations made from (4) and (5)
%IMPORTANT NOTE:
%the elements of matrices a and c (see below) are expressions based on (4) and (5) and part
%of program which calculates the partial derivatives (from line %Find the first partial...
%to the line %Find the inital...)
%However the elements itself are computed outside the function since the computation itself
%(for instance via MATLAB eval function: a(2*i-1,1)=eval(f(1));a(2*i-1,2)=eval(f(2)); etc.
%c(2*i-1)=sk(i,1)-eval(u);c(2*i)=sk(i,2)-eval(v);)is only time consuming and unnecessary.
%Thus the mentioned part of the program has only educational/historical purpose and 
%can be excluded for practical purposes

for k=1:ite  %k-th iteartion
   %Form matrices a and c
   for i=1:m	%i-th point
      X=pk(i,1);Y=pk(i,2);Z=pk(i,3);
      %first row of the i-th point; contribution of (4) equation
      a(2*i-1,1)=(-X*L6*L9^2-X*L6*L11^2+X*L10*L5*L9+X*L10*L7*L11+Y*L5*L10^2+Y*L5*L11^2-Y*L9*L6*L10-Y*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11);
      a(2*i-1,2)=(-X*L7*L9^2-X*L7*L10^2+X*L11*L5*L9+X*L11*L6*L10+Z*L5*L10^2+Z*L5*L11^2-Z*L9*L6*L10-Z*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11);
      a(2*i-1,3)=(L5*L10^2+L5*L11^2-L9*L6*L10-L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11);
      a(2*i-1,4)=(L4*L10^2+L4*L11^2+X*L2*L10*L9+X*L3*L11*L9+L2*Y*L10^2+L2*Y*L11^2+L3*Z*L10^2+L3*Z*L11^2)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)^2*(L11^2+L10^2+L9*X*L10^2+L9*X*L11^2+L10^3*Y+L10*Y*L11^2+L11*Z*L10^2+L11^3*Z);
      a(2*i-1,5)=(-X*L2*L9^2-X*L2*L11^2+X*L3*L11*L10-L2*Y*L9*L10-L3*Z*L9*L10-L4*L9*L10)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)^2*(-L9^2*X*L10-L10^2*Y*L9-L11*Z*L9*L10-L9*L10);
      a(2*i-1,6)=(-X*L3*L9^2-X*L3*L10^2+X*L2*L10*L11-L2*Y*L9*L11-L3*Z*L9*L11-L4*L9*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)^2*(-L9^2*X*L11-L10*Y*L9*L11-L11^2*Z*L9-L9*L11);
      a(2*i-1,7)=0;
      a(2*i-1,8)=(-2*X*L2*L6*L9-2*X*L3*L7*L9+X*L2*L10*L5+X*L3*L11*L5-L2*Y*L6*L10-L2*Y*L7*L11-L3*Z*L6*L10-L3*Z*L7*L11-L4*L6*L10-L4*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)^2*(X*L5*L10^2+X*L5*L11^2-2*L9*X*L6*L10-2*L9*X*L7*L11-L10^2*Y*L6-L10*Y*L7*L11-L11*Z*L6*L10-L11^2*Z*L7-L6*L10-L7*L11);
      a(2*i-1,9)=(2*L4*L5*L10-2*X*L3*L7*L10+X*L2*L5*L9+X*L2*L7*L11+X*L3*L11*L6+2*L2*Y*L5*L10-L2*Y*L9*L6+2*L3*Z*L5*L10-L3*Z*L9*L6-L4*L9*L6)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)^2*(2*L5*L10+2*L9*X*L5*L10-L9^2*X*L6+3*L10^2*Y*L5+Y*L5*L11^2-2*L10*Y*L9*L6-Y*L9*L7*L11+2*L11*Z*L5*L10-L11*Z*L9*L6-L9*L6);
      a(2*i-1,10)=(2*L4*L5*L11-2*X*L2*L6*L11+X*L2*L10*L7+X*L3*L5*L9+X*L3*L6*L10+2*L2*Y*L5*L11-L2*Y*L9*L7+2*L3*Z*L5*L11-L3*Z*L9*L7-L4*L9*L7)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11)^2*(2*L5*L11+2*L9*X*L5*L11-L9^2*X*L7+2*L10*Y*L5*L11-L10*Y*L9*L7+Z*L5*L10^2+3*L11^2*Z*L5-Z*L9*L6*L10-2*L11*Z*L9*L7-L9*L7);
      %second row of the i-th point; contribution of (5) equation
      a(2*i,1)=0;
      a(2*i,2)=0;
      a(2*i,3)=0;
      a(2*i,4)=X/(L9*X+L10*Y+L11*Z+1);
      a(2*i,5)=Y/(L9*X+L10*Y+L11*Z+1);
      a(2*i,6)=Z/(L9*X+L10*Y+L11*Z+1);
      a(2*i,7)=1/(L9*X+L10*Y+L11*Z+1);
      a(2*i,8)=-(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1)^2*X;
      a(2*i,9)=-(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1)^2*Y;
      a(2*i,10)=-(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1)^2*Z;
      %analogicaly for c matrice
      c(2*i-1)=sk(i,1)-(L4*L5*L10^2+L4*L5*L11^2-X*L2*L6*L9^2-X*L2*L6*L11^2-X*L3*L7*L9^2-X*L3*L7*L10^2+X*L2*L10*L5*L9+X*L2*L10*L7*L11+X*L3*L11*L5*L9+X*L3*L11*L6*L10+L2*Y*L5*L10^2+L2*Y*L5*L11^2-L2*Y*L9*L6*L10-L2*Y*L9*L7*L11+L3*Z*L5*L10^2+L3*Z*L5*L11^2-L3*Z*L9*L6*L10-L3*Z*L9*L7*L11-L4*L9*L6*L10-L4*L9*L7*L11)/(L5*L11^2+L5*L10^2+L9*X*L5*L10^2+L9*X*L5*L11^2-L9^2*X*L6*L10-L9^2*X*L7*L11+L10^3*Y*L5+L10*Y*L5*L11^2-L10^2*Y*L9*L6-L10*Y*L9*L7*L11+L11*Z*L5*L10^2+L11^3*Z*L5-L11*Z*L9*L6*L10-L11^2*Z*L9*L7-L9*L6*L10-L9*L7*L11);
      c(2*i)=sk(i,2)-(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1);
   end
   c=c';c=c(:); %regrouping in one column
   b=a\c; %10 MDLT parameters of the k-the iteration
   
   % Prepare the estimates for a new iteration
   L2=b(1)+L2;L3=b(2)+L3;L4=b(3)+L4;L5=b(4)+L5;L6=b(5)+L6;
   L7=b(6)+L7;L8=b(7)+L8;L9=b(8)+L9;L10=b(9)+L10;L11=b(10)+L11;
   % Calculate L1 based on equation (1)and 'save' the parameters of the k-th iteration
   dlt(k,:)=[eval(L1) L2 L3 L4 L5 L6 L7 L8 L9 L10 L11],
   disp('Number of iterations performed'),k
   clear a b c
end
b=dlt;%return all sets of 11 DLT parameters for all iterations
