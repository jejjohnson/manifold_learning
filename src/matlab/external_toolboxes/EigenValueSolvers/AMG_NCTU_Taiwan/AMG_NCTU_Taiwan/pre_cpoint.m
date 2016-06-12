function [c,f,u]=pre_cpoint(A,s,st,lamda)

[row,col]=size(A);
c=sparse(1,col);
f=sparse(1,col);
u=[1:row];
cnum=1;
l=0;
m=0;
start=1;
while (any(u)),
    [max_value,max_lamda]=max(lamda);
    c(cnum)=max_lamda; %%% C = C union {vi}
    cnum=cnum+1;
    u(max_lamda)=0;    %%% U= U - {vi}
    lamda(max_lamda)=-9999999;
    temp=find(st(max_lamda,:)~=0);
    sit_col=size(temp,2);
    clear temp
    temp=find(s(max_lamda,:)~=0);
    si_col=size(temp,2);
    clear temp
    %sit_col=count_nonzero(st(max_lamda,:));
    %si_col=count_nonzero(s(max_lamda,:));
    add_to_f=0;
    for j=1:sit_col,
        mask=[];
        mask=find(u==st(max_lamda,j));      
        if(~isempty(mask)) %%% (S_i)T and U ~= empty
            %%%step 4
            l=l+1;
            f(l)=st(max_lamda,j);  %%% F=F union {vj=st(max_lamda,j)}
            u(f(l))=0;             %%% U=U-{vj}
            lamda(f(l))=-9999999;
            %%% step 5
            sj_col=count_nonzero(s(f(l),:));
            for k=1:sj_col,
                mask=[];
                mask=find(u==s(f(l),k));      
                if(~isempty(mask))  %%%(S_j) and U ~= empty
                    lamda(s(f(l),k))=lamda(s(f(l),k))+1; %%% zk=zk+1
                end
            end    
        end
    end
    %%% step 6
    for k=1:si_col,
        mask=[];
        mask=find(u==s(max_lamda,k));      
        if(~isempty(mask))
            lamda(s(max_lamda,k))=lamda(s(max_lamda,k))-1; %%%zj=zj-1
        end
    end
end %%% end while

      
         
