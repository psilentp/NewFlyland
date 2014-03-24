% File used to test different routines for 3D calibration of caneras.

data1=[];

[file, pname] = uigetfile('*.mat','Calibration image data file selection');
cd(pname);
load(file);

% % remove first surface
% F = F(73:end,:);
% L = L(73:end,:,:);
% 
% % remove middle surface
% F(73:144,:) = [];
% L(73:144,:,:) = [];
% 
% % remove last surface
% F = F(1:144,:);
% L = L(1:144,:,:);

% world coord (45deg tilt)
Pts = F;

% camera coordinates
Cam1 = L(:,:,1);
Cam2 = L(:,:,2);
Cam3 = L(:,:,3);

prompt = {'Specify routine (DLT or MDLT)'};
dlg_title = 'Calibration routine:';
num_lines = 1;
def = {'DLT'};
Method = inputdlg(prompt, dlg_title, num_lines, def);
if length(char(Method{1})) == 3
    Meth = 1;
elseif length(char(Method{1})) == 4
    Meth = 2;
else
    disp('Not an acceptable routine.');
    return
end

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

% Saving of coefficients to allow use of Peter Maddens Digimat
fLego=[];
fCamPoints=[];
[fn,pn] = uiputfile('*.mat', 'Save CAMERA COEFFICIENT FILE:');
save([pn, fn], 'CamCoef', 'fLego', 'fCamPoints');
