
%% analysis of results:

% number of neurons: 29 - 80 - 100 - 4
num_sims = 5;
for i=1:num_sims
    name = ['cond_1_',num2str(i),'_file.mat'];
    load(name)
    overall_accuracy_cond_1(i,:) = output.accuracy;
    all_mi_cond_1(i,:) = mutual_info_analysis_allo(output);

end

ydata_1 = all_mi_cond_1(:);
xdata_1= ones(length(ydata_1),1);

overall_accuracy_cond_1(4,:) =[];


% number of neurons: 29 - 100 - 80 - 4
for i=1:num_sims
    name = ['cond_4_',num2str(i),'_file.mat'];
    load(name)
    overall_accuracy_cond_2(i,:) = output.accuracy;
        all_mi_cond_2(i,:) = mutual_info_analysis_allo(output);
end
overall_accuracy_cond_2(2,:)=[];
ydata_2 = all_mi_cond_2(:);
xdata_2= 2.*ones(length(ydata_2),1);

% number of neurons: 29 - 100 - 100 - 4
for i=1:num_sims
    name = ['cond_5_',num2str(i),'_file.mat'];
    load(name)
    overall_accuracy_cond_3(i,:) = output.accuracy;
            all_mi_cond_3(i,:) = mutual_info_analysis_allo(output);
end
ydata_3 = all_mi_cond_3(:);
xdata_3= 3.*ones(length(ydata_3),1);

hold on 
plot_distribution(1:10,overall_accuracy_cond_1,'r')
plot_distribution(1:10,overall_accuracy_cond_2,'b')
plot_distribution(1:10,overall_accuracy_cond_3,'g')
xlabel('Epoch')
ylabel('Total Reward')
%% 
% number of neurons: 29 - 80 - 100 - 4
num_sims = 5;
for i=1:num_sims
    name = ['cond_6_',num2str(i),'_file.mat'];
    load(name)
    overall_accuracy_cond_6(i,:) = output.accuracy;
                all_mi_cond_6(i,:) = mutual_info_analysis_allo(output);
end
ydata_6 = all_mi_cond_6(:);
xdata_6= 2.*ones(length(ydata_6),1);

overall_accuracy_cond_6(3,:)=[];
hold on 
plot_distribution(1:10,overall_accuracy_cond_1,'r')
plot_distribution(1:10,overall_accuracy_cond_6,'b')
xlabel('Epoch')
ylabel('Total Reward')



%% Mutual information:

% Visulize for single agent:
figure
subplot(3,2,1)
neuron_idx =  100;
imagesc(reshape(all_activity(:,neuron_idx),5,5))
colormap("hot")
c = colorbar;
c.Label.String = 'Activity';
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:5)
xticklabels(1:5)
yticks(1:5)
yticklabels(1:5)
xlabel('position')
ylabel('position')
clim([-1 4])

subplot(3,2,2)
neuron_idx =  100;
plot(1:n_sensory_cue, all_activity_sensory(:,neuron_idx),'k',LineWidth=2)
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:4)
xticklabels({'light port 1','light port 2','odor 1','odor 2'})
xlabel('position')
ylabel('activity')


subplot(3,2,3)
neuron_idx =  1;
imagesc(reshape(all_activity(:,neuron_idx),5,5))
colormap("hot")
c = colorbar;
c.Label.String = 'Activity';
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:5)
xticklabels(1:5)
yticks(1:5)
yticklabels(1:5)
xlabel('position')
ylabel('position')
clim([-1 4])

subplot(3,2,4)
neuron_idx =  1;
plot(1:n_sensory_cue, all_activity_sensory(:,neuron_idx),'k',LineWidth=2)
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:4)
xticklabels({'light port 1','light port 2','odor 1','odor 2'})
xlabel('position')
ylabel('activity')


subplot(3,2,5)
neuron_idx =  88;
imagesc(reshape(all_activity(:,neuron_idx),5,5))
colormap("hot")
c = colorbar;
c.Label.String = 'Activity';
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:5)
xticklabels(1:5)
yticks(1:5)
yticklabels(1:5)
xlabel('position')
ylabel('position')
clim([-1 4])

subplot(3,2,6)
neuron_idx =  88;
plot(1:n_sensory_cue, all_activity_sensory(:,neuron_idx),'k',LineWidth=2)
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:4)
xticklabels({'light port 1','light port 2','odor 1','odor 2'})
xlabel('position')
ylabel('activity')



figure
subplot(3,2,1)
neuron_idx =  5;
imagesc(reshape(all_activity(:,neuron_idx),5,5))
colormap("hot")
c = colorbar;
c.Label.String = 'Activity';
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:5)
xticklabels(1:5)
yticks(1:5)
yticklabels(1:5)
xlabel('position')
ylabel('position')
clim([-1 4])

subplot(3,2,2)
neuron_idx =  5;
plot(1:n_sensory_cue, all_activity_sensory(:,neuron_idx),'k',LineWidth=2)
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:4)
xticklabels({'light port 1','light port 2','odor 1','odor 2'})
xlabel('position')
ylabel('activity')


subplot(3,2,3)
neuron_idx =  6;
imagesc(reshape(all_activity(:,neuron_idx),5,5))
colormap("hot")
c = colorbar;
c.Label.String = 'Activity';
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:5)
xticklabels(1:5)
yticks(1:5)
yticklabels(1:5)
xlabel('position')
ylabel('position')
clim([-1 4])

subplot(3,2,4)
neuron_idx =  6;
plot(1:n_sensory_cue, all_activity_sensory(:,neuron_idx),'k',LineWidth=2)
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:4)
xticklabels({'light port 1','light port 2','odor 1','odor 2'})
xlabel('position')
ylabel('activity')


subplot(3,2,5)
neuron_idx =  7;
imagesc(reshape(all_activity(:,neuron_idx),5,5))
colormap("hot")
c = colorbar;
c.Label.String = 'Activity';
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:5)
xticklabels(1:5)
yticks(1:5)
yticklabels(1:5)
xlabel('position')
ylabel('position')
clim([-1 4])

subplot(3,2,6)
neuron_idx =  7;
plot(1:n_sensory_cue, all_activity_sensory(:,neuron_idx),'k',LineWidth=2)
title('MI:', num2str(mutual_info(neuron_idx)))
xticks(1:4)
xticklabels({'light port 1','light port 2','odor 1','odor 2'})
xlabel('position')
ylabel('activity')


% average over agents:

hold on
jitterAmount = 0.1;
jitterValuesX_1 = 2*(rand(size(xdata_1))-0.5)*jitterAmount;   % +/-jitterAmount max
scatter(xdata_1+jitterValuesX_1, ydata_1);
jitterValuesX_2 = 2*(rand(size(xdata_2))-0.5)*jitterAmount;   % +/-jitterAmount max
scatter(xdata_2+jitterValuesX_2, ydata_2);
jitterValuesX_3 = 2*(rand(size(xdata_3))-0.5)*jitterAmount;   % +/-jitterAmount max
scatter(xdata_3+jitterValuesX_3, ydata_3);
xlim([0 4])
ylabel('MI')
xticks([1 2 3])
xticklabels({'Cond 1','Cond 2','Cond 3'})


hold on
jitterAmount = 0.1;
jitterValuesX_1 = 2*(rand(size(xdata_1))-0.5)*jitterAmount;   % +/-jitterAmount max
scatter(xdata_1+jitterValuesX_1, ydata_1);
jitterValuesX_6 = 2*(rand(size(xdata_6))-0.5)*jitterAmount;   % +/-jitterAmount max
scatter(xdata_6+jitterValuesX_6, ydata_6);
xlim([0 3])
ylabel('MI')
xticks([1 2])
xticklabels({'Cond 1','Cond 4'})





