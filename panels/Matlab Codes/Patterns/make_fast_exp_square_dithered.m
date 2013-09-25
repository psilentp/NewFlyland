%make_4_wide_stripe_pattern_12.m
% 12 panels in a circle -> 96 columns, 1 panel high -> 8 rows
%InitPat = [ones(1,92)*7 zeros(1,4)];
%pattern form: Elevation, Azmuth ,x depth, y depth
%%
fs = filesep;

pathmap = containers.Map();
pathmap('Windows_NT') = 'Pcontrol_paths_win.mat';
load(pathmap(getenv('os')));

%%
%pixels in actual pattern
azmuth_pix = 8*12;
elev_pix = 8*3;
pattern.gs_val = 3;
pattern.x_num = azmuth_pix*(2^pattern.gs_val);%same as image Azm
pattern.y_num = 12;
pattern.num_panels = 12*3;

pattern.row_compression = 0;
ExpanPole = pattern.x_num/2;

%shape of temp image pre dither and crop
ImgSize = pattern.x_num;
%dimensions for crop pre dither
ImgAzm = azmuth_pix*(2^pattern.gs_val);
ImgElev = elev_pix*(2^pattern.gs_val);
ImgCenter = [floor(ImgAzm/2) floor(ImgAzm/2)];

%what is needed to transform from large to small image via interp.
DegPerPx = 360/azmuth_pix;
DegPerXstp = 360/pattern.x_num;
ZeroPxOffset = 45/DegPerPx - 4;

Pats = ones(elev_pix, azmuth_pix, pattern.x_num, pattern.y_num+1)*7;

%we draw a single frame pre crop into img_mat
img_mat = ones(pattern.x_num-1,pattern.x_num-1)*7;
%temp_mat holds the movie for a single y value
movie_mat = ones(elev_pix,azmuth_pix,pattern.x_num)*7;

%Expand the pattern on all sides 45 deg
for j = 1:floor(90/DegPerXstp)
    fprintf('.')
    img_mat(ExpanPole-j:ExpanPole+j,ExpanPole-j:ExpanPole+j) = 0;
    tmp_mat = floor(imresize(img_mat,[azmuth_pix azmuth_pix],'box'));
    azmuth_pix/2-floor(elev_pix/2);
    azmuth_pix/2+floor(elev_pix/2);
    movie_mat(:,:,j) = tmp_mat(azmuth_pix/2-floor(elev_pix/2)+1:azmuth_pix/2+floor(elev_pix/2),:);
end
%%
% 
%create the set of stimuli -180 to + 150 in 30deg increments
fprintf('\n')
for ExpPol = [1:12;-180:30:150]
    fprintf('.')
    shift = floor(ExpPol(2)/DegPerPx);
    Pats(:,:,:,ExpPol(1)+1) = floor(circshift(movie_mat,[0,ZeroPxOffset+shift,0]));
end

pattern.Pats = Pats;
pattern.Panel_map = [36 32 28 35 31 27 34 30 26 33 29 25;
                     24 20 16 23 19 15 22 18 14 21 17 13;
                     12 8  4  11 7  3  10 6  2  9  5  1];
                   
pattern.BitMapIndex = process_panel_map(pattern);
 
pattern.data = Make_pattern_vector(pattern);

directory_name = pattern_path;
str = [directory_name fs 'Pattern_dithered_expansion_sq'];

save(str, 'pattern');

%% Functions
update_freq = 90;
trial_duration = 5.0;
times = -1*trial_duration:1/update_freq:-0.001;
trial_tail = 0.4; %400ms
expan_fun = @(lv_ratio,t) floor(atand(lv_ratio ./ t)/DegPerXstp);

func_path = function_path;

hold on
func = [expan_fun(-0.005,times), ones(1,trial_tail*update_freq)*expan_fun(-0.005,-0)];
save([ func_path fs 'position_function_expan_5ms_sq.mat'], 'func');
plot(func)

func = [expan_fun(-0.02,times), ones(1,trial_tail*update_freq)*expan_fun(-0.02,-0)];
save([ func_path fs 'position_function_expan_20ms_sq.mat'], 'func');
plot(func)

func = [expan_fun(-0.05,times), ones(1,trial_tail*update_freq)*expan_fun(-0.05,-0)];
save([ func_path fs 'position_function_expan_50ms_sq.mat'], 'func');
plot(func)

func = [expan_fun(-0.1,times), ones(1,trial_tail*update_freq)*expan_fun(-0.1,-0)];
save([ func_path fs 'position_function_expan_100ms_sq.mat'], 'func');
plot(func)

func = [expan_fun(-0.2,times), ones(1,trial_tail*update_freq)*expan_fun(-0.2,-0)];
save([ func_path fs 'position_function_expan_500ms_sq.mat'], 'func');
plot(func)