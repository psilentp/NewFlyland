% rename calibration image files (*.tif) for checkerboard recognition prog
warning off

loc=cd;
files = dir('*.bmp');
infofile = dir('*.cih');

old_info = infofile(1).name;
new_info = [loc(end-12:end),old_info(end-3:end)];
%         movefile(old_name,new_name);
java.io.File(old_info).renameTo(java.io.File(new_info));
% dos(['rename "' old_info '" "' new_info '"']);

if isempty(files)==0
    for i=1:length(files)
        old_name = files(i).name;
        new_name = [loc(end-12:end),old_name(end-9:end)];
        
%         copyfile(old_name,new_name);
%         movefile(old_name,new_name);
        java.io.File([old_name]).renameTo(java.io.File([new_name]));
%         dos(['rename "' old_name '" "' new_name '"']);
        
        counter=length(files)-i
    end
end
        
    