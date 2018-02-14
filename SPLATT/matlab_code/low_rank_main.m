%% align arbitrary # of networks with the same size
clear all;clc
alpha=0.5; 
AUCvec=[];
timevec=[];
topK=1000 %% rank K
n=100; %% size of networks
density=0.1; %% density of networks(approximate)
noise=0.1; %% noise from ancester network
net_num=5; %% # of networks
%% generate random networks and return eigenvalues, eigenvectors
%% and normalized adjecency matrics
[Q,lam,S]= generate_rand_net(n,net_num,density,noise);
%% Generate initial tensor of size n^net_num
% offdiag_percent=0.01;%% sampling a percentage of off diagnal entries
[subs,vals]=ini_spar_tensor(n,net_num);%% return index and values of non-zeros in tensor
F=sptensor(subs,vals,n*ones(1,net_num)); %% construct tensor
%% Select topK eigenvalues and eigenvectors using greedy method
[lam_mat,q]=greedy_select_topK(lam,Q,n,topK,net_num);
lam_k=prod(lam_mat,1)';
lam_k=1./(1-alpha*lam_k);

t1=cputime;
%% run low rank algorithm
f=zeros(topK,1);

% Use tensor toolbox to perform tensor-times-vector with all but the first mode.
%MT = mttkrp(F, q, 1);
%% Incorporate the first mode.
%for rank=1:topK
%  f(rank) = q{1}(:,rank)' * MT(:,rank);
%end

for row=1:topK
    if mod(row,200)==0
        row
    end
    q_now=q{net_num};    
    Q=q_now(subs(:,1),row);  
    for idx=net_num-1:-1:1
        q_now=q{idx};    
        Q=[Q,q_now(subs(:,net_num-idx+1),row)];    
    end
    f(row)= sum(prod(Q,2).*vals);
end

f=f.*lam_k(1:topK);
pred_vals=zeros(length(vals),1);

for num=1:length(vals)
    if mod(num,10^5)==0
        num
    end
    q_now=q{1};
    i=subs(num,1);
    row=q_now(i,:);
    for idx=2:net_num
        q_now=q{idx};
        i=subs(num,idx);
        row=row.*q_now(i,:);
    end
    pred_vals(num)=row*f;
end
t=cputime-t1;
scores=pred_vals(n/2+1:end); %% only look at the entries corresponding
% to those nonzero entris in initial tensor

%% plot AUC
labels=zeros(length(scores),1);
labels(1:n/2)=1;
[X,Y,T,AUC] = perfcurve(labels,scores,1);
AUC
AUCvec=[AUCvec,AUC];
timevec=[timevec,t];

write_output([subs vals], lam_k, q, [subs pred_vals], f);


