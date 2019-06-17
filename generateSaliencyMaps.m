% This script generates saliency map images for all .jpg images in the path folder
% Outputs will be saved as imagename_saliency.jpg
% Assumed is that the drfi_matlab library is installed as published on 
% https://github.com/playerkk/drfi_matlab (retrieved 17 June 2019)

path = 'D:\CS\CS4180 Deep Learning\Project\images\';
files = dir(fullfile(path,'\*.jpg'));
addpath(genpath('.'));
para = makeDefaultParameters;

for i = 1:length(files)
    t = tic;
    image = imread( [path files(i).name] );
    smap = drfiGetSaliencyMap( image, para );
    imwrite(smap,[path files(i).name(1:end-4) '_saliency' files(i).name(end-3:end)]);
    time_cost = toc(t);
    fprintf( 'Saved saliency image %.f out of %.f taking %f seconds', i, length(files), time_cost );
end