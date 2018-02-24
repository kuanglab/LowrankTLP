%% simulation_data.mat: simulation dataset generated according to section 7.1 of:
%% Scalable Label Propagation for Multi-relational Learning on Tensor Product Graph 
%% arXiv:1802.07379 [cs.LG]
% S: a cell array stores five 1000 X 1000 normalized sparse graphs

% Y0: a sparse tensor stores multi-relations. The format of Y0 is:
% a1 b1 c1 d1 e1 val1
% a2 b2 c2 d2 e2 val2

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 d1 e1
% a2 b2 c2 d2 e2

% labels_test: labels of subs_test for evaluation purpose

clear all;clc
load('simulation_data.mat');
Q = cell(1,length(S));
lam = cell(1,length(S));
graphsizes = zeros(1,length(S));
for netid = 1:length(S)
[Q{netid},lam{netid}] = eig(S{netid});
lam{netid} = diag(real(lam{netid}));
graphsizes(netid) = size(S{netid},1);
end
k = 2*10^4;
alpha = 0.1;
Y = LowrankTLP_opt1(lam,Q,k,graphsizes,alpha,Y0,subs_test); % run LowrankTLP algorithm
[~,~,~,AUC] = perfcurve(labels_test,Y,1);
disp(['AUC = ',num2str(AUC)]);


