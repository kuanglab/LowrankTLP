% Acar, Evrim, et al. "Scalable tensor factorizations for incomplete data." Chemometrics and Intelligent Laboratory Systems 106.1 (2011): 41-56.
%% inputs
% Y0: a sparse tensor with dimension graphsizes storing the known
% multi-relations. The i-th mode of Y0 matches with the W{i}. 
% The format of Y0 is:
% a1 b1 c1 val1
% a2 b2 c2 val2
% where (ai bi ci) is a tupple of indices in tensor Y0 and vali is the 
% corresponding value.

% rank_k: CP rank

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 
% a2 b2 c2

% stopcrit: stopping criteria

% MaxIers: the maximum number of ADAM iterations
%% outputs
% pred_vals: a vector of prediction scores of the queried multi-relations in subs_test
function pred_vals=CP_W(Y0,rank_k,subs_test,stopcrit,MaxIters)
n = size(Y0);
net_num = length(n);
Q=cell(net_num,1);
for netid=1:net_num
    Q{netid}=randn(n(netid),rank_k);
end
subs0 = Y0.subs;
vals0 = Y0.vals;

% initial_gradients
G = CP_W_get_GD(Q,subs0,vals0,rank_k,net_num,n);
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
G = CP_W_get_GD(Q,subs0,vals0,rank_k,net_num,n);

G_now=0;
for netid=1:net_num
    G_now=G_now+sum(sum(G{netid}.^2));
end
G_now=sqrt(G_now);

ratio=G_now/G_ini;
if mod(iter,100)==0
    disp('CP-W');
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

function G = CP_W_get_GD(Q,subs0,vals0,rank_k,net_num,n)
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
for netid = 1:net_num
G{netid} = -mttkrp(X_comb,Q,netid);
end
end




