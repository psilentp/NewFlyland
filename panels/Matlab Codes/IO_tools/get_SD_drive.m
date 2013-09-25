function [SD_drive] = get_SD_drive
% [SD_drive] = get_SD_drive
% function takes no inputs and returns the letter of the SD drive
% it also checks to see that the SD drive is indeed removable
pathmap = containers.Map()
pathmap('Windows_NT') = 'Pcontrol_paths_win.mat'
load(pathmap(getenv('os')));
%load('Pcontrol_paths.mat');

if exist('myPCCfg.mat','file')
    disp 'now here'
    load([controller_path '\myPCCfg'],'-mat');
    if isfield(myPCCfg, 'SDDrive')
        SD_drive = myPCCfg.SDDrive;
        defaultDrive = {SD_drive};
        defaultDrive = {'G'};
        disp 'here'
    else
        defaultDrive = {'G'};
    end
end
    

    
    
    
    
    
    
    
   
SD_drive = userInputSDDrive(defaultDrive);


myPCCfg.SDDrive = SD_drive;
save([controller_path '\myPCCfg'], 'myPCCfg');
