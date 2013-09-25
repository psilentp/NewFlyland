%make_4_wide_stripe_pattern_12.m
% 12 panels in a circle -> 96 columns, 1 panel high -> 8 rows
%InitPat = [ones(1,92)*7 zeros(1,4)];
%pattern form: Elevation, Azmuth ,x depth, y depth
fs = filesep

azmuth_pix = 96;
elev_pix = 1;
pattern.gs_val = 3;
pattern.x_num = azmuth_pix*(2^pattern.gs_val);
pattern.y_num = 12;
pattern.num_panels = 12;

pattern.row_compression = 1;
MatCenter = pattern.x_num/2;
ExpanPole = MatCenter;
DegPerPx = 360/azmuth_pix;
DegPerXstp = 360/pattern.x_num;
ZeroPxOffset = 45/DegPerPx - 4;

Pats = ones(elev_pix, azmuth_pix, pattern.x_num, pattern.y_num+1)*7;

img_mat = ones(elev_pix,pattern.x_num)*7;
temp_mat = ones(elev_pix,azmuth_pix,pattern.x_num)*7;

%Expand the pattern on both sides 45 deg
for j = 1:floor(90/DegPerXstp)
    img_mat(:,MatCenter-j:ExpanPole+j) = 0;
    temp_mat(:,:,j) = floor(imresize(img_mat,[elev_pix azmuth_pix],'box'));
end
%%
% 
%create the set of stimuli -180 to + 150 in 30deg increments
for ExpPol = [1:12;-180:30:150]
    shift = floor(ExpPol(2)/DegPerPx);
    Pats(:,:,:,ExpPol(1)+1) = floor(circshift(temp_mat,[0,ZeroPxOffset+shift,0]));
end

pattern.Pats = Pats;
pattern.Panel_map = [12 8 4 11 7 3 10 6 2 9 5 1];
pattern.BitMapIndex = process_panel_map(pattern);
pattern.data = Make_pattern_vector(pattern);

directory_name = pattern_path;
str = [directory_name fs 'Pattern_dithered_expansion'];
save(str, 'pattern');

%% Functions
update_freq = 500;
trial_duration = 5.0;
times = -1*trial_duration:1/update_freq:-0.001;
expan_fun = @(lv_ratio,t) floor(atand(lv_ratio ./ t)/DegPerXstp);

func_path = function_path;

%hold on
func = [expan_fun(-0.005,times), ones(1,100)*expan_fun(-0.005,-0)];
save([ func_path fs 'position_function_expan_5ms.mat'], 'func');
%plot(func)

func = [expan_fun(-0.02,times), ones(1,100)*expan_fun(-0.02,-0)];
save([ func_path fs 'position_function_expan_20ms.mat'], 'func');
%plot(func)

func = [expan_fun(-0.05,times), ones(1,100)*expan_fun(-0.05,-0)];
save([ func_path fs 'position_function_expan_50ms.mat'], 'func');
%plot(func)

func = [expan_fun(-0.1,times), ones(1,100)*expan_fun(-0.1,-0)];
save([ func_path fs 'position_function_expan_100ms.mat'], 'func');
%plot(func)

func = [expan_fun(-0.2,times), ones(1,100)*expan_fun(-0.2,-0)];
save([ func_path fs 'position_function_expan_500ms.mat'], 'func');
%plot(func)
