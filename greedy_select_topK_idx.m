%% inputs
% lam: a cell array with dimention n, where n is the number of graphs.
% the i-th entry lam{i} stores the eigenvalues of the i-th graph.

% topK: the rank of the approximated tensor product graph Sk.

% net_num: number of graphs

% alpha: parameter of label propagation
%% outputs
% x: the topK selected eigenvalues.
% q: the indices of selected eigenpairs of each graph
function [x,q]=greedy_select_topK_idx(lam,topK,net_num,alpha)
Indmat=cell(net_num,1);
x=lam{1};
for i=2:net_num
    disp(['Algorithm 1: Select Eigenvalues sequentially ... graph: ',num2str(i)]);
    x1=lam{i};
    x=kron(x,x1);
    [a,b]=sort(x,'descend');
    
    if i<net_num
        if 2*topK<=length(x)
    x=[a(1:topK);a(end-topK+1:end)];
    idx=[b(1:topK);b(end-topK+1:end)];
        else
            x=a;
            idx=b;
        end
    else 
    y=alpha*abs(x)./(1-alpha*x);
    [~,b]=sort(y,'descend');
    idx=b(1:topK);
    x=x(idx);
    end
    L2=length(x1);
    mat=zeros(length(idx),2);
    for ind=1:length(idx)
        [a,b]=returnIndex2(L2,idx(ind));
        mat(ind,:)=[a,b];
    end
    Indmat{i}=mat;
end
mat=Indmat{net_num};
left=mat(:,1);
right=mat(:,2);
selectmat(:,net_num)=right;
for idx=net_num-1:-1:2
    mat=Indmat{idx};
    left1=mat(:,1);
    right1=mat(:,2);
    ind=right1(left);
    selectmat(:,idx)=ind;
    left=left1(left);
end
selectmat(:,1)=left;
q=cell(net_num,1);

for idx=net_num:-1:1
    q_now=zeros(1,topK);
    for id=1:topK
    q_now(id)=selectmat(id,idx);
    end
    q{idx}=q_now;
end
end

function [i,j]=returnIndex2(L2,ind) 
i=ceil(ind/L2);
j=mod(ind,L2);
if j==0
    j=L2;
end
end
