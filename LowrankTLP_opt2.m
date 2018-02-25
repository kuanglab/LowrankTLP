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
function Y=LowrankTLP_opt2(lam,Q,k,r,graphsizes,alpha,R,subs_test)
net_num=length(graphsizes);
[lam_k,id_cell]=greedy_select_topK_idx(lam,k,net_num,alpha);
q={};
for i=1:net_num
    q{i}=Q{i}(:,id_cell{i});
end
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


%% Compression step
thre=.001;pow=2;
Q=SymNMF(dataMat,sum(graphsizes),r,thre, pow);
A{1}=Q(1:graphsizes(1),:);
for netid=2:net_num
    A{netid}=Q(sum(graphsizes(1:netid-1))+1:sum(graphsizes(1:netid)),:);
end
v=q{1}'*A{1};
for i=2:net_num
    v=v.*(q{i}'*A{i});
end
v=sum(v,2); 
lam_k=1./(1-alpha*lam_k)-1;
v=v.*lam_k;

%% Expansion step

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
    


