outputVideo = VideoWriter('video.avi', 'Uncompressed AVI');
outputVideo.FrameRate = 48;
open(outputVideo)
workingDir = '/home/mark/code/udacity/CarND-Behavioral-Cloning-P3'
imageNames = dir(fullfile(workingDir,'video1','*.jpg'));
%imageNames = dir('/home/mark/code/udacity/CarND-Behavioral-Cloning-P3/video1');
imageNames = {imageNames.name}';

for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,'video1',imageNames{ii}));
   writeVideo(outputVideo,img)
end

close(outputVideo)