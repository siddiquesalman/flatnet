



% Start with a folder and get a list of all subfolders.
% Finds and prints names of all PNG, JPG, and TIF images in
% that folder and all of its subfolders.
% Similar to imageSet() function in the Computer Vision System Toolbox: http://www.mathworks.com/help/vision/ref/imageset-class.html
clear all
clc;    % Clear the command window.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
% calib=load('flatcam_prototype2_calibdata.mat'); % load calibration data
lmbd = 3e-4; % regularization parameter
nr=2;
% Define a starting folder.
start_path = fullfile('/mnt/data1/sk135/Amplitude Mask/dataset/orig');
folder1 = fullfile('/home/salman/Flatcam_separable_CNN/Train/INET_UNSEEN_OR_Newer/');
folder2 = fullfile('/home/salman/Flatcam_separable_CNN/Val/INET_UNSEEN_OR_Newer/');

% Ask user to confirm or change.
topLevelFolder = uigetdir(start_path);
if topLevelFolder == 0
    return;
end
% Get list of all subfolders.
allSubFolders = genpath(topLevelFolder);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
while true
    [singleSubFolder, remain] = strtok(remain, ':');
    if isempty(singleSubFolder)
        break;
    end
    listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames)
vc =1;
tc=1;
h=0;
label = [];
p =1;
totl_img =0;
% Process all image files in those folders.
arr=[89,91,302,666,921,454,455,484,951,971];
for k = 1 : numberOfFolders
    % Get this folder and print it out.
    thisFolder = listOfFolderNames{k};
    fprintf('Processing folder %s\n', thisFolder);
    k
    noval = 1;
    indx=1;
   
    % Get PNG files.
    filePattern = sprintf('%s/*.png', thisFolder);
    baseFileNames = dir(filePattern);
    % Add on TIF files.
    filePattern = sprintf('%s/*.tif', thisFolder);
    baseFileNames = [baseFileNames; dir(filePattern)];
    % Add on JPG files.
    filePattern = sprintf('%s/*.JPEG', thisFolder);
    baseFileNames = [baseFileNames; dir(filePattern)];
    numberOfImageFiles = length(baseFileNames);
    totl_img = totl_img + numberOfImageFiles;
    % Now we have a list of all files in this folder.
    
    if numberOfImageFiles >= 1
        % Go through all those image files.

        for f = 1 : numberOfImageFiles
            fullFileName = fullfile(thisFolder, baseFileNames(f).name);
            class_str = strrep(thisFolder,'/home/salman/Downloads/','');
            class = str2num(class_str);
            fprintf('Processing image file %s\n', fullFileName);
%             im = imread(fullFileName);
            %im = reconstruct_flatcam(im,calib,lmbd);
            if ismember(k-1,arr)
                files2{vc,1} = fullFileName(18:end);
                vc = vc+1;
            else
                files1{tc,1} = fullFileName(18:end);
                tc=tc+1;
            end
            
            
            label = [label class ] ;
            
        end
        
    else
        fprintf('Folder %s has no image files in it.\n', thisFolder);
    end
end
fid_w1 = fopen('train_orig_ilsvrc_flatcam.txt', 'w');
fid_w2 = fopen('val_orig_ilsvrc_flatcam.txt', 'w');
fprintf(fid_w1, '%s\n', files1{:});
fprintf(fid_w2, '%s\n', files2{:});
fclose(fid_w1);
fclose(fid_w2);