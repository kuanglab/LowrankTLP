%% Scalable Label Propagation for Multi-relational Learning on Tensor Product Graph 
%% arXiv:1802.07379 [cs.LG]
% lam: a cell array with dimention n, where n is the number of graphs.
% the i-th entry lam{i} stores the eigenvalues of the i-th graph.

% Q: a cell array with dimention n, where n is the number of graphs.
% the i-th entry Q{i} stores the eigenvectors of the i-th graph.

% k: the rank of the approximated tensor product graph Sk.

% r: the rank of symNMF. We suggest to choose the k which captures
% 90 percent variance using PCA.

% graphsizes: a n-D vector, whose i-th entry is the size the i-th graph.

% alpha: parameter of label propagation

% R: a n by n cell array, where n is the number of graphs.
% Rij stores the pairwise relations between graph i and j where i<j.

% subs_test: a matrix storing the indices of the queried multi-relations in
% the format
% a1 b1 c1 
% a2 b2 c2

%% outputs: 
% Y the predicted labels of the queried multi-relations
function Y=LowrankTLP_Task2(W,k,graphsizes,alpha,R,subs_test)
net_num=length(graphsizes);
[Q,lam]=Eig_pair(W, net_num);
[lam_k,id_cell]=greedy_select_topK_idx(lam,k,net_num,alpha);
%% Compression step
disp('LowrankTLP: Compression ...');
q=cell(1,net_num);
for i=1:net_num
    q{i}=Q{i}(:,id_cell{i});
end
A = obtain_CP_form(R,net_num,graphsizes); % get the CP-form Y0
v=q{1}'*A{1};
for i=2:net_num
    v=v.*(q{i}'*A{i});
end
v=sum(v,2); 
lam_k=1./(1-alpha*lam_k)-1;
v=v.*lam_k;

%% Prediction step
disp('LowrankTLP: Prediction ...');

Y0=zeros(size(subs_test,1),1);
for i=1:size(subs_test,1)
    row=A{1}(subs_test(i,1),:);
    for netid=2:net_num
        row=row.*A{netid}(subs_test(i,netid),:);
    end
    Y0(i)=sum(row);
end

Y=zeros(length(Y0),1);
for num=1:length(Y0)
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
Y=(1-alpha)*(Y+Y0);
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

