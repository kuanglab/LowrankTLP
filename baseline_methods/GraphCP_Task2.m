% Narita, Atsuhiro, et al. "Tensor factorization using auxiliary information." Data Mining and Knowledge Discovery 25.2 (2012): 298-324.
%% Inputs:
% R: a n by n cell array, where n is the number of graphs.
% Rij stores the pairwise relations between graph i and j.

% graphsizes: a n-d vector, whose i-th entry is the size of i-th graph

% W: a n-d cell array whose i-th entry W{i} stores the adjacency matrix of the i-th undirected graph.

% lambda: graph hyperparameter

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 
% a2 b2 c2

% stopcrit: stopping criteria

% MaxIers: the maximum number of ADAM iterations


function pred_vals=GraphCP_Task2(R,graphsizes,W,lambda,subs_test,stopcrit,MaxIters)
net_num = length(W);
Q0 = obtain_CP_form(R,net_num,graphsizes); % get the CP-form Y0
D=cell(net_num,1);
for netid=1:net_num
    D{netid} = diag(sum(W{netid},2));
end
% initial_gradients
Q=Q0;
G = GraphCP_Task2_getGD(Q0,Q,W,D,lambda,net_num);
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
m{netid} = zeros(graphsizes(netid),size(Q{netid},2));
end
v = m; % second moment
t=0;
ratio=1;

% run ADAM
for iter=1:MaxIters
    if ratio<stopcrit
        break;
    end
G = GraphCP_Task2_getGD(Q0,Q,W,D,lambda,net_num);

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


function G = GraphCP_Task2_getGD(Q0,Q,W,D,lambda,net_num)
G=cell(net_num,1);
rank_k=size(Q{1},2);
for netid=1:net_num
    M1=ones(rank_k,rank_k);
    for factid=1:net_num
        if factid~=netid
            M1=M1.*(Q{factid}'*Q{factid});
        end
    end
    cum=ones(rank_k,rank_k);
    for factid=1:net_num
        if factid~=netid
            cum=cum.*(Q0{factid}'*Q{factid});
        end
    end
    M3=Q0{netid}*cum;
    cum1=1;
    cum2=1;
    for factid=1:net_num
        if factid~=netid
            cum1=cum1*trace(Q{factid}'*D{factid}*Q{factid});
            cum2=cum2*trace(Q{factid}'*W{factid}*Q{factid});
        end
    end
    L=cum1*D{netid}-cum2*W{netid};
    G{netid}=Q{netid}*M1+lambda*L*Q{netid}-M3;
end
end

function A = obtain_CP_form(R,net_num,graphsizes)
dataMat=cell(net_num,net_num);
for i=1:net_num
    for j=1:net_num
        if i<j
        dataMat{i,j}=R{i,j}; 
        elseif i>j
            dataMat{i,j}=R{j,i}';
        else
            dataMat{i,j}=zeros(graphsizes(i),graphsizes(j));
        end
    end
end
dataMat=cell2mat(dataMat);
[~, ~, pcvars] = pca(full(dataMat));
varvec=cumsum(pcvars./sum(pcvars) * 100);
ind=find(varvec<90);
ind=ind(end)+1;
r=ind; 

thre=.001;pow=2;
Q=SymNMF(dataMat,sum(graphsizes),r,thre, pow); 
A=cell(1,net_num);
A{1}=Q(1:graphsizes(1),:);
for netid=2:net_num
    A{netid}=Q(sum(graphsizes(1:netid-1))+1:sum(graphsizes(1:netid)),:);
end
end


function Q=SymNMF(dataMat,n,rank_k,thre,pow)
Q = full(2 * sqrt(mean(mean(dataMat)) / rank_k) * rand(n, rank_k));
for iter=1:10000
    Q_old=Q;
    Q=Q.*((dataMat*Q+10^-20)./(Q*(Q'*Q)+10^-20)).^(1/pow);
    res=sqrt(sum(sum((Q-Q_old).^2)))/sqrt(sum(sum(Q.^2)));
    disp('res is: ');
    disp(res);
    if res < thre
        break
    end
end
end



