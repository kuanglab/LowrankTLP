%% simulation_data.mat: simulation dataset generated according to section 7.1 of:
%% Scalable Label Propagation for Multi-relational Learning on Tensor Product Graph 
%% arXiv:1802.07379 [cs.LG]
% install MATLAB tensor toolbox before running
clear all;clc
load Data/simulation_data
% W: a cell array stores five 1000 X 1000 graphs

% Y0: a sparse tensor stores multi-relations. The format of Y0 is:
% a1 b1 c1 d1 e1 val1
% a2 b2 c2 d2 e2 val2

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 d1 e1
% a2 b2 c2 d2 e2

% labels_test: test labels of subs_test for evaluation purpose

alpha=0.1; % graph hyperparameter
k=2*10^4;  % TPG rank

pred_vals = LowrankTLP_Task1(W,k,alpha,Y0,subs_test); % run LowrankTLP
[~,~,~,AUC] = perfcurve(labels_test,pred_vals,1);
disp(['AUC = ', num2str(AUC)]);

cd baseline_methods

pred_vals = ApproxLink(W,k,alpha,Y0,subs_test); % run ApproxLink
[~,~,~,AUC] = perfcurve(labels_test,pred_vals,1);
disp(['AUC = ', num2str(AUC)]);

%% CP based methods perform randomly for the extremely sparse Y0
subs_train = Y0(:,1:end-1);
vals_train = Y0(:,end);
graphsizes = zeros(1,length(W));
for i=1:length(W)
    graphsizes(i) = size(W{i},1);
end
Y0=sptensor(subs_train,vals_train,graphsizes); %% construct the initial tensor
r=10; % CP rank
stopcrit = 10^-3; % ADAM stopping criteria
maxIters = 1000; % max number of iterations


pred_vals=GraphCP_W(Y0,W,alpha,r,subs_test,stopcrit,maxIters); % run GraphCP_W
[~,~,~,AUC] = perfcurve(labels_test,pred_vals,1);
disp(['AUC = ', num2str(AUC)]);

pred_vals=GraphCP_Task1(Y0,W,alpha,r,subs_test,stopcrit,maxIters); % run GraphCP
[~,~,~,AUC] = perfcurve(labels_test,pred_vals,1);
disp(['AUC = ', num2str(AUC)]);

pred_vals=CP_W(Y0,r,subs_test,stopcrit,maxIters); % run CP_W
[~,~,~,AUC] = perfcurve(labels_test,pred_vals,1);
disp(['AUC = ', num2str(AUC)]);

