%make_8_wide_stripe_pattern_12.m
% 12 panels in a circle -> 96 columns, 1 panel high -> 8 rows
InitPat = [zeros(8*4,2) ones(8*4,94)];

pattern.x_num = 96;
pattern.y_num = 1;
pattern.num_panels = 12*4;
pattern.gs_val = 1;
pattern.row_compression = 0;

Pats = zeros(8*4,96 , pattern.x_num, pattern.y_num);
Pats(:,:,1,1) = InitPat;

for j = 2:96
    Pats(:,:,j,1) = ShiftMatrix(Pats(:,:,j-1,1), 1, 'r', 'y'); 
end

pattern.Pats = Pats;
pattern.Panel_map = [48 44 40 47 43 39 46 42 38 45 41 37;
                     36 32 28 35 31 27 34 30 26 33 29 25;
                     24 20 16 23 19 15 22 18 14 21 17 13;
                     12 8  4  11 7  3  10 6  2  9  5  1];
                 
pattern.BitMapIndex = process_panel_map(pattern);
pattern.data = Make_pattern_vector(pattern);
directory_name = 'C:\Users\psilentp\Documents\NewFlyland\panels\Matlab Codes\Patterns';
%directory_name = 'C:\Users\psilentp\Dropbox\MATLAB\panels\Matlab Codes\Patterns';
str = [directory_name '\Pattern_fixation_2_wide_12Pan'];
save(str, 'pattern');