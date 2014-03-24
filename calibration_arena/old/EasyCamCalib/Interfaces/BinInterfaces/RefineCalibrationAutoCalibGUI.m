function outstruct = RefineCalibrationAutoCalibGUI(ImageData,DIVMODEL)

%% Configurations
global input
N = [1:length(ImageData)];
RECOMPUTETRANSFORM = 0;     %Put to 1 if you want to recompute the Plane2eye transform after initializing the intrinsics


%% Initialize the intrinsics to refine   
StackAratio = []; StackSkew = []; StackCenterx = []; StackCentery = []; StackFocal = []; StackQsi = [];
for i=N
   StackAratio = [StackAratio ImageData(i).FinalCalib.aratio];
   StackSkew = [StackSkew ImageData(i).FinalCalib.skew];
   StackCenterx = [StackCenterx ImageData(i).FinalCalib.center(1)];
   StackCentery = [StackCentery ImageData(i).FinalCalib.center(2)];
   StackFocal = [StackFocal ImageData(i).FinalCalib.focal];
   StackQsi = [StackQsi ImageData(i).FinalCalib.qsi];
end
aratio = median(StackAratio);
skew = median(StackSkew);
center(1) = median(StackCenterx);
center(2) = median(StackCentery);
focal = median(StackFocal);
qsi = median(StackQsi);
eta = focal/sqrt(-qsi);
fprintf('Initial Intrinsics: qsi=%f, focal=%f, eta=%f, center=(%f,%f), a=%f, s=%f \n', qsi, focal, eta, center, aratio, skew);


%% Recompute the Plane2Eye transform according to the new intrinsics
if RECOMPUTETRANSFORM
    auxStruct.aratio = aratio;
    auxStruct.skew = skew;
    auxStruct.center(1) = center(1);
    auxStruct.center(2) = center(2);
    auxStruct.focal = focal;
    auxStruct.qsi = qsi;
    auxStruct.eta = eta;
    j=1;
    for i=N
        NewT (:,:,j) = ComputeRigidTransform(ImageData(i).PosPlane,ImageData(i).PosImage,auxStruct);
        j=j+1;
    end
end

%% Set the inputs
input.N = length(N);
input.DIVMODEL = DIVMODEL;
input.aratio = aratio;
input.skew = skew;
input.NPTS = 0;                                             
for i=N
   input.NPTS = [input.NPTS length(ImageData(i).PosImage)]; %Number of points in each image
end

%% Set the parameters to refine
% [qsi;focal;centerx;centery]
% For each image put the Plane2Eye transform in the form [[Theta Alpha Beta]';[tx ty tz]']
if DIVMODEL==1
    initvalue = [qsi;focal;center(1);center(2);];
else
    initvalue = [0.0;qsi;focal;center(1);center(2);]; %%The first distortion parameter in the taylor expansion is initialized to 0.0
end
for i=1:input.N
    if RECOMPUTETRANSFORM
        T = NewT(:,:,i);
    else
        T = ImageData(i).FinalCalib.T;
    end
    [Theta w] = R2Euler(T(1:3,1:3));
    Alpha = acos(w(3));
    Beta = atan2(w(2),w(1));
    initvalue = [initvalue;Theta;Alpha;Beta;T(1:3,4)];
end


%% Stack the experimental data
xdata = [];
ydata = [];
for i=N
   xdata = [xdata ImageData(i).PosPlane(1:2,:)]; 
   ydata = [ydata ImageData(i).PosImage(1:2,:)];   
end
ydata = reshape(ydata,2*length(ydata),1);


%% Do the Refinement
% options = optimset('MaxFunEval',15000,'Display','iter','MaxIter',1000,'LevenbergMarquardt','on','TolFun',1E-8,'Algorithm','levenberg-marquardt','Jacobian','on');
options = optimset('MaxFunEval',15000,'Display','iter','MaxIter',1000,'LevenbergMarquardt','on','TolFun',1E-8,'Algorithm','levenberg-marquardt');
[x resnorm residual dummy info s e] = lsqcurvefit(@FitFunctionIntrinsics,initvalue,xdata,ydata,[],[],options);


%% Get the results
jump=0;
for i=1:input.N 
    ImageData(i).OptimCalib=[]; %clear the current structure 
    ImageData(i).OptimCalib.aratio = aratio;
    ImageData(i).OptimCalib.skew = skew;
    if DIVMODEL == 1
        numintrinsics=4;
        offset = numintrinsics+jump;
        ImageData(i).OptimCalib.qsi = x(1);
        ImageData(i).OptimCalib.focal = x(2);
        ImageData(i).OptimCalib.center(1) = x(3);
        ImageData(i).OptimCalib.center(2) = x(4);
    else
        numintrinsics=5;
        offset = numintrinsics+jump;
        ImageData(i).OptimCalib.qsi0 = x(1);
        ImageData(i).OptimCalib.qsi = x(2);
        ImageData(i).OptimCalib.focal = x(3);
        ImageData(i).OptimCalib.center(1) = x(4);
        ImageData(i).OptimCalib.center(2) = x(5);
    end
    ImageData(i).OptimCalib.eta = ImageData(i).OptimCalib.focal/sqrt(-ImageData(i).OptimCalib.qsi);
    ImageData(i).OptimCalib.K=[ImageData(i).OptimCalib.aratio*ImageData(i).OptimCalib.focal ImageData(i).OptimCalib.skew*ImageData(i).OptimCalib.focal ImageData(i).OptimCalib.center(1);...
            0 ImageData(i).OptimCalib.aratio^-1*ImageData(i).OptimCalib.focal ImageData(i).OptimCalib.center(2);...
            0 0 1];
    ImageData(i).OptimCalib.Keta = [ImageData(i).OptimCalib.aratio*ImageData(i).OptimCalib.eta ImageData(i).OptimCalib.skew*ImageData(i).OptimCalib.eta ImageData(i).OptimCalib.center(1);0 ImageData(i).OptimCalib.aratio^-1*ImageData(i).OptimCalib.eta ImageData(i).OptimCalib.center(2); 0 0 1];
    Theta = x(1+offset);
    Alpha = x(2+offset); 
    Beta = x(3+offset);       
    t = [x(4+offset);x(5+offset);x(6+offset)];                                
    w = [sin(Alpha)*cos(Beta);sin(Alpha)*sin(Beta);cos(Alpha)];
    R = Euler2R(Theta,w);
    ImageData(i).OptimCalib.T = [R t; 0 0 0 1];
    if DIVMODEL == 1
        ImageData(i).OptimCalib.ReProjError = ReProjectionError(ImageData(i).OptimCalib, ImageData(i).PosPlane, ImageData(i).PosImage);
        ImageData(i).OptimCalib=orderfields(ImageData(i).OptimCalib,{'eta','aratio','skew','focal','qsi','center','Keta','K','T','ReProjError'});
    else
        ImageData(i).OptimCalib.ReProjError = ReProjectionError2ndOrder(ImageData(i).OptimCalib, ImageData(i).PosPlane, ImageData(i).PosImage);
        ImageData(i).OptimCalib=orderfields(ImageData(i).OptimCalib,{'eta','aratio','skew','focal','qsi0','qsi','center','Keta','K','T','ReProjError'});
    end
    jump=6*(i); 
end

outstruct = ImageData;