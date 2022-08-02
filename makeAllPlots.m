

figure(1)
subplot(2,2,1)
a =reshape(maxQ(1:25),5,5);
hold on 
imagesc(reshape(maxQ(1:25),5,5))
[X,Y] = meshgrid(1:1:5);
[dX,dY] = gradient(a,1);
quiver(X,Y,dX,dY,'k','LineWidth',2)
hold off
title('Max Q - odor top')

subplot(2,2,2)
a =reshape(maxQ(26:50),5,5);
hold on 
imagesc(reshape(maxQ(26:50),5,5))
[X,Y] = meshgrid(1:1:5);
[dX,dY] = gradient(a,1);
quiver(X,Y,dX,dY,'k','LineWidth',2)
hold off
title('Max Q - odor bottom')

subplot(2,2,3)
a =reshape(maxQ(51:75),5,5);
hold on 
imagesc(reshape(maxQ(51:75),5,5))
[X,Y] = meshgrid(1:1:5);
[dX,dY] = gradient(a,1);
quiver(X,Y,dX,dY,'k','LineWidth',2)
hold off
title('Max Q - reward right')

subplot(2,2,4)
a =reshape(maxQ(76:end),5,5);
hold on 
imagesc(reshape(maxQ(76:end),5,5))
[X,Y] = meshgrid(1:1:5);
[dX,dY] = gradient(a,1);
quiver(X,Y,dX,dY,'k','LineWidth',2)
hold off
title('Max Q - reward left')


figure(2)
subplot(2,2,1)
imagesc(reshape(state_occupancy1(1:25),5,5))
title('State Occupancy - context 1: odor top')

subplot(2,2,2)
imagesc(reshape(state_occupancy1(26:50),5,5))
title('State Occupancy - context 2: odor bottom')

subplot(2,2,3)
imagesc(reshape(state_occupancy1(51:75),5,5))
title('State Occupancy - context 1: reward right')
subplot(2,2,4)
imagesc(reshape(state_occupancy1(76:100),5,5))
colorbar
title('State Occupancy - context 2: reward left')

figure(3)

plot(mean(numStepsCon1,1))
title('Number of Steps to Reward')
xlabel('Session')
ylabel('Number of Steps')


