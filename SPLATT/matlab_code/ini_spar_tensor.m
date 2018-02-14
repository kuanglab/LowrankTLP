function [subs,vals]=ini_spar_tensor(n,net_num)
subs=ones(1,net_num);
for i=2:n
    subs=[subs;i*ones(1,net_num)];
end
vals=ones(size(subs,1),1);
vals(n/2+1:end)=0.1;
% offdiag_percent=0.01;
randsub=zeros(n/2,net_num);
for i=1:size(randsub,1)
    randsub(i,:)=randperm(n,net_num);
end
subs=[subs;randsub];
vals=[vals;0.1*ones(size(randsub,1),1)];
end
