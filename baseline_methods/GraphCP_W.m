% Narita, Atsuhiro, et al. "Tensor factorization using auxiliary information." Data Mining and Knowledge Discovery 25.2 (2012): 298-324.
%% inputs
% Y0: a sparse tensor with dimension graphsizes storing the known
% multi-relations. The i-th mode of Y0 matches with the W{i}. 
% The format of Y0 is:
% a1 b1 c1 val1
% a2 b2 c2 val2
% where (ai bi ci) is a tupple of indices in tensor Y0 and vali is the 
% corresponding value.

% W: a n-d cell array whose i-th entry W{i} stores the adjacency matrix of the i-th undirected graph.

% lambda: graph hyperparameter

% rank_k: CP rank

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 
% a2 b2 c2

% stopcrit: stopping criteria

% MaxIers: the maximum number of ADAM iterations
%% outputs
% pred_vals: a vector of prediction scores of the queried multi-relations in subs_test
function pred_vals=GraphCP_W(Y0,W,lambda,rank_k,subs_test,stopcrit,MaxIters)
n = size(Y0);
net_num = length(W);
Q=cell(net_num,1);
for netid=1:net_num
    Q{netid}=randn(n(netid),rank_k);
end
D=cell(net_num,1);
for netid=1:net_num
    D{netid} = diag(sum(W{netid},2));
end
subs0 = Y0.subs;
vals0 = Y0.vals;

% initial_gradients
G = GraphCP_W_get_GD(Q,W,D,subs0,vals0,rank_k,lambda,net_num,n);
G_ini=0;
for netid=1:net_num
    G_ini=G_ini+sum(sum(G{netid}.^2));
end
G_ini=sqrt(G_ini);  

% ADAM parameters
alpha=0.001;
Beta1=0.9;
Beta2=0.999;
epsilon=10^-8;
m = cell(1,net_num); % first moment
for netid = 1:net_num
m{netid} = zeros(n(netid),rank_k);
end
v = m; % second moment
t=0;
ratio=1;

% run ADAM
for iter=1:MaxIters
    if ratio<stopcrit
        break;
    end
G = GraphCP_W_get_GD(Q,W,D,subs0,vals0,rank_k,lambda,net_num,n);

G_now=0;
for netid=1:net_num
    G_now=G_now+sum(sum(G{netid}.^2));
end
G_now=sqrt(G_now);

ratio=G_now/G_ini;
if mod(iter,100)==0
    disp('GraphCP-W');
    disp(['iter: ',num2str(iter)]);
    disp(['ratio: ',num2str(ratio)]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=t+1;
mold=m;
vold=v;
for netid = 1:net_num
m{netid}=Beta1*mold{netid}+(1-Beta1)*G{netid};
v{netid}=Beta2*vold{netid}+(1-Beta2)*G{netid}.^2;
m_hat=m{netid}/(1-Beta1^t);
v_hat=v{netid}/(1-Beta2^t);
Q{netid}=Q{netid}-alpha*m_hat./(sqrt(v_hat)+epsilon);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

pred_vals=zeros(size(subs_test,1),1);
for i=1:size(subs_test,1)
    cum=Q{1}(subs_test(i,1),:);
    for j=2:net_num
        cum=cum.*Q{j}(subs_test(i,j),:);
    end    
    pred_vals(i)=sum(cum);
end
end

function G = GraphCP_W_get_GD(Q,W,D,subs0,vals0,rank_k,lambda,net_num,n)
vals = zeros(size(subs0,1),1);
for row = 1:length(vals)
    cumvec = ones(1,rank_k);
    for netid = 1:net_num
        index = subs0(row,netid);
        cumvec = cumvec.*Q{netid}(index,:);
    end
    vals(row) = sum(cumvec);
end
vals_comb = vals0-vals;
X_comb = sptensor(subs0,vals_comb,n);
G = cell(size(n));
traceDvec = zeros(net_num,1);
traceWvec = zeros(net_num,1);
for netid = 1:net_num
    traceDvec(netid) = trace(Q{netid}'*D{netid}*Q{netid});
    traceWvec(netid) = trace(Q{netid}'*W{netid}*Q{netid});
end
for netid = 1:net_num
cum1=1;
cum2=1;
for factid=1:net_num
    if factid~=netid
       cum1=cum1*traceDvec(factid);
       cum2=cum2*traceWvec(factid);
    end
end
L=cum1*D{netid}-cum2*W{netid};
G{netid} = -mttkrp(X_comb,Q,netid) + lambda*L*Q{netid};
end
end




