state_mov1 = state;
    states = zeros(1,25);

movie_dur = min(length(state_mov1),1000);

for i=1:movie_dur
    states=states*0.80;
    if mod(state_mov1(i),25) ~=0
    states(mod(state_mov1(i),25))=1;
    else
        states(25) =1;
    end
    states = reshape(states, 5,5);
    img = imagesc(states);
    F(i) = getframe(gcf); 
    drawnow

end

writerObj = VideoWriter('myVideo.avi');
  writerObj.FrameRate = 10;


% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(F)
    % convert the image to a frame
    frame = F(i) ;    
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);