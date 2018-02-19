function Y=LowrankTLP_opt1(lam,Q,k,n,alpha,Y0,subs_test)
net_num=length(n);
subs_train=Y0(:,1:net_num);
subs_train=fliplr(subs_train);   
vals_train=Y0(:,end);
[lam_k,id_cell]=greedy_select_topK_idx(lam,k,net_num,alpha);
q={};
for i=1:net_num
    q{i}=Q{i}(:,id_cell{i});
end
%% Compression step
disp('Algorithm 2: Compression ...');
v=zeros(k,1);
for row=1:k
    q_now=q{net_num};    
    Q=q_now(subs_train(:,1),row);  
    for idx=net_num-1:-1:1
        q_now=q{idx};    
        Q=[Q,q_now(subs_train(:,net_num-idx+1),row)];    
    end
    v(row)= sum(prod(Q,2).*vals_train);
end
lam_k=1./(1-alpha*lam_k)-1;
v=v.*lam_k;
disp('Compresion end!');
%% Expansion step
disp('Algorithm 2: Expansion ...');
Y=zeros(size(subs_test,1),1);
for num=1:length(Y)
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
disp('Expansion end!');
Y=(1-alpha)*Y;
end
    


