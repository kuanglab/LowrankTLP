% Raymond, Rudy, and Hisashi Kashima. "Fast and scalable algorithms for semi-supervised link prediction on static and dynamic graphs." Joint european conference on machine learning and knowledge discovery in databases. Springer, 
% Berlin, Heidelberg, 2010.
%% inputs
% W: a cell array with dimention n, where n is the number of graphs.
% the i-th entry W{i} stores the adjacency matrix of the i-th undirected graph.
% k: the rank of the TPG rank.
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
function pred_vals = ApproxLink(W,k,alpha,Y0,subs_test)
net_num = length(W);
subs_train = Y0(:,1:net_num);
vals_train = Y0(:,end);
subs_test = fliplr(subs_test);
subs_train = fliplr(subs_train);
K=ceil(nthroot(k,net_num));
[Q,lam,graphsizes] = ApproxGraph(W,K,net_num);
disp('ApprxLink Compression ...');
F=sptensor(subs_train,vals_train,fliplr(graphsizes)); %% construct tensor
Q=fliplr(Q);
X = tenones(K*ones(1,net_num));
subs=find(X);
X=sptensor(K*ones(1,net_num));
X=tensor(X);
for i=1:size(subs,1)  
    subs_row=subs(i,:);
    multi_vec=cell(1,length(subs_row));
    for j=1:length(subs_row)
        idx=subs_row(j);
        Q_now=Q{j};
        multi_vec{j}=Q_now(:,idx);
    end
    X(subs_row)= ttv(F,  multi_vec, (1:net_num));
end
F=X;
clear X subs1
F=tenmat(F,(1:net_num),'t');
Atsize=F.tsize;
Ardims=F.rdims;
Acdims=F.cdims;
F=F.data;
lam_now=lam{1};
for i=2:net_num
    lam_now=kron(lam_now,lam{i});
end
lam=1./(1-alpha*lam_now)-1;
F=F.*lam';
F = tenmat(F,Ardims,Acdims,Atsize); 
F = tensor(F);
disp('ApprxLink Prediction ...');
pred_vals=zeros(size(subs_test,1),1);
for i=1:size(subs_test,1)
    subs_row=subs_test(i,:);
    multi_vec=cell(1,length(subs_row));
    for j=1:length(subs_row)
        idx=subs_row(j);
        Q_now=Q{j};
        multi_vec{j}=Q_now(idx,:)';
    end
    pred_vals(i)= ttv(F,  multi_vec, (1:net_num));
end

function [Q,lam,graphsizes] = ApproxGraph(W,K,net_num)
disp('Approxmating each graph ...');
Q=cell(1,net_num);
lam=Q;
graphsizes=ones(1,net_num);
for netid=1:net_num
graphsizes(netid) = size(W{netid},1);
[V,L]=eig(W{netid});
L=real(diag(L));
[~,idx]=sort(abs(L),'descend');
V=V(:,idx(1:K));
L=L(idx(1:K));
G=V*diag(L)*V';
D=sum(G,2);
D(D~=0)=(D(D~=0)).^-(0.5); 
S=diag(D)*G*diag(D);
[V,L]=eig(S);
L=real(diag(L));
[~,idx]=sort(abs(L),'descend');
V=V(:,idx(1:K));
L=L(idx(1:K));
Q{netid}=V;
lam{netid}=L;
end

