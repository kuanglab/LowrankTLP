%% Scalable Label Propagation for Multi-relational Learning on Tensor Product Graph 
%% arXiv:1802.07379 [cs.LG]
%% inputs
% W: a n-d cell array whose i-th entry W{i} stores the adjacency matrix of the i-th undirected graph.

% k: the rank of S_k.

% alpha: graph hyperparameter

% Y0: a sparse tensor with dimension graphsizes storing the known
% multi-relations. The i-th mode of Y0 matches with the W{i}. 
% The format of Y0 is:
% a1 b1 c1 val1
% a2 b2 c2 val2
% where (ai bi ci) is a tupple of indices in tensor Y0 and vali is the 
% corresponding value.

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 
% a2 b2 c2
%% outputs
% pred_vals: a vector of prediction scores of the queried multi-relations
% in subs_test
function Y=LowrankTLP_Task1(W,k,alpha,Y0,subs_test)

net_num=length(W);
[Q,lam]=Eig_pair(W, net_num);
subs_train=Y0(:,1:net_num);
subs_train=fliplr(subs_train);   
vals_train=Y0(:,end);
[lam_k,id_cell]=greedy_select_topK_idx(lam,k,net_num,alpha); % Algorithm 1 
q=cell(1,net_num);
for i=1:net_num
    q{i}=Q{i}(:,id_cell{i});
end
%% Compression step
disp('LowrankTLP: Compression ...');
v=zeros(k,1);
for row=1:k   
    Q=zeros(size(subs_train,1),net_num);
    col=0;
    for idx=net_num:-1:1
        col=col+1;
        q_now=q{idx};    
        Q(:,col)=q_now(subs_train(:,net_num-idx+1),row);    
    end
    v(row)= sum(prod(Q,2).*vals_train);
end
lam_k=1./(1-alpha*lam_k)-1;
v=v.*lam_k;
%% Prediction step
disp('LowrankTLP: Prediction ...');
Y=zeros(size(subs_test,1),1);
for num=1:length(Y)
    q_now=q{1};
    i=subs_test(num,1);
    row=q_now(i,:);
    for idx=2:net_num
        q_now=q{idx};
        i=subs_test(num,idx);
        row=row.*q_now(i,:);
    end
    Y(num)=row*v;
end
disp('Expansion end!');
Y=(1-alpha)*Y;
end

function [Q,lam]=Eig_pair(W, net_num)
S=cell(net_num,1); % graph normalization
for netid = 1:net_num
    D=sum(W{netid},2);
    D(D~=0)=(D(D~=0)).^-(0.5); 
    S{netid}=diag(D)*W{netid}*diag(D);
end
Q = cell(1,length(S));
lam = Q;
for netid = 1:net_num
[Q{netid},lam{netid}] = eig(S{netid}); % computing eigenvalues 
lam{netid} = diag(real(lam{netid}));
end
end
    


