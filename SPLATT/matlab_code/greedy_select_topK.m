function [lam_mat,q]=greedy_select_topK(lam,Q,n,topK,net_num)
Indmat=cell(net_num,1);
x=lam{1};
for i=2:net_num
    x1=lam{i};
    x=kron(x,x1);
    [a,b]=sort(x,'descend');
    if i<net_num
    x=[a(1:topK);a(end-topK+1:end)];
    idx=[b(1:topK);b(end-topK+1:end)];
    else 
     x=a(1:topK);
    idx=b(1:topK);
    end
    L2=length(x1);
    mat=[[],[]];
    for ind=1:length(idx)
        [a,b]=returnIndex2(L2,idx(ind));
        mat=[mat;[a,b]];
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

lam_mat=zeros(net_num,topK);
for idx=net_num:-1:1
    Q_now=Q{idx};
    q_now=zeros(n,topK);
    lam_now=lam{idx};
    for id=1:topK
    lam_mat(idx,id)=lam_now(selectmat(id,idx));
    q_now(:,id)=Q_now(:,selectmat(id,idx));
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
