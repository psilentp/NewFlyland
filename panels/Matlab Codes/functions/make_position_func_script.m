% make_func_script.m
% make and save some position function files

func_path = 'C:\Users\psilentp\Documents\MATLAB\panels\Matlab Codes\functions\';
% make a 20 position peak to peak 0.25 Hz position sine wave
func = round(10*make_sine_wave_function(20, 50, 0.25));
save([ func_path 'position_function_sine_025Hz_20_pp.mat'], 'func');

% make a 50 position peak to peak 0.1 Hz position sine wave
func = round(25*make_sine_wave_function(20, 50, 0.1));
save([ func_path 'position_function_sine_01Hz_50_pp.mat'], 'func');

% make a 50 position peak to peak 0.2 Hz position sine wave, mod 8
func = sign_mod(round(25*make_sine_wave_function(20, 50, 0.2)), 8);
save([ func_path 'position_function_sine_02Hz_50_pp_moded_8.mat'], 'func');

% make a 50 position peak to peak 0.2 Hz position sine wave, mod 8
func = sign_mod(round(25*make_sine_wave_function(20, 50, 0.2)), 16);
save([ func_path 'position_function_sine_02Hz_50_pp_moded_16.mat'], 'func');

% make a 100 position peak to peak 0.1 Hz position sine wave
func = round(50*make_sine_wave_function(20, 50, 0.1));
save([ func_path 'position_function_sine_01Hz_100_pp.mat'], 'func');

% make a 100 position peak to peak 0.1 Hz position sine wave
func = round(50*make_sine_wave_function(40, 50, 0.1));
save([ func_path 'position_function_sine_01Hz_100_pp_var.mat'], 'func');

% make a 20 position peak to peak 0.25 Hz position sine wave
func = round(10*make_sine_wave_function(50, 50, 0.25));
save([ func_path 'position_function_sine_025Hz_20_pp_var.mat'], 'func');

% make a linear function
func = 1:100;
func(50:100) = 0;
save([ func_path 'position_function_linear_stop.mat'], 'func');

