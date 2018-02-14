function[Q,lam,S]= generate_rand_net(n,net_num,density,noise)
W=zeros(n,n);
for i=1:n
R = sprand(n,1,density);
W(:,i)=R;
end
W=W+W';
W=double(logical(W));
for i=1:n
    W(i,i)=0;
end
%%
Q=cell(net_num,1);
lam=cell(net_num,1);
S=cell(net_num,1);
for net_idx=1:net_num
W1=W;
mat=zeros(ceil(n^2/2*noise),2);
for i=1:ceil(n^2/2*noise)
    c1=randperm(n);
    c1=c1(1);
    c2=randperm(n);
    c2=c2(1);
    mat(i,:)=[c1,c2];
end
v=(mat(:,1)==mat(:,2));
ind=find(v==1);
mat(ind,:)=[];
for i=1:size(mat,1)
    W1(mat(i,1),mat(i,2))=1-W1(mat(i,1),mat(i,2));
    W1(mat(i,2),mat(i,1))=W1(mat(i,1),mat(i,2));
end
D1=diag(sqrt(1./sum(W1,2)));
S1=D1*W1*D1;
[Q1,lam1]=eig(S1);
lam1=diag(real(lam1));
Q{net_idx}=Q1;
lam{net_idx}=lam1;
S{net_idx}=S1;
end
end