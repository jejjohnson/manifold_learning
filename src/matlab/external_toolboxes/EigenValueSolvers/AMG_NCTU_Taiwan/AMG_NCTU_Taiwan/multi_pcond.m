function [MPC]=multi_pcond( A_stack,  lpindx, m_cindx, l_cindx,level,w,pc_type)

%%% pc_type=1 => Jacobi, pc_type=2 => Gauss-Seidel, pc_type=3 => LU %%%
%%% w: over-relaxation parameter for SOR

prnum= 1; 
MPC=sparse(lpindx(level+1)-1,prnum*(lpindx(2)-1));

if(pc_type==3)
    prnum=1;
end
iind=[];jind=[];kvalue=[];
for i=1:level,
    actual_block_num=0;
    knum=lpindx(i+1)-lpindx(i);
    PK=sparse(knum,knum);
    PK=A_stack(1:knum,lpindx(i):lpindx(i+1)-1);
    pindx=m_cindx(i,1:knum);
    pindx1=[1:knum];
    
    if(i<=level-1)
        cknum=lpindx(i+2)-lpindx(i+1);
        cpindx=l_cindx(i+1,1:cknum);
        fcount=0;
        fpindx=[];
        for j=1:knum,
            mask=[];
            mask=find(cpindx==pindx1(j));
            if(isempty(mask))
                fcount=fcount+1;
                fpindx(fcount)=pindx1(j);
            end
        end      
    else
        cknum=knum;
        cpindx=[1:knum];
    end
    
    clear aa bb cc
    for k=1:prnum,

        if(pc_type<=3)
            if(pc_type==0)
                %MPC(lpindx(i):lpindx(i+1)-1,(k-1)*knum+1:k*knum)=speye(knum);
                II=speye(knum);
                [aa,bb,cc]=find(II);
            end
            if(pc_type==1)
                JPC=1/w.*spdiags(diag(PK,0),0,knum,knum);
                [aa,bb,cc]=find(JPC);
                %MPC(lpindx(i):lpindx(i+1)-1,(k-1)*knum+1:k*knum)=JPC;%(iper(k,:),iper(k,:));
                clear JPC;
            end
            if(pc_type==2)                    
                GSPC=sparse(tril(PK,-1))+1/w.*spdiags(spdiags(PK,0),0,knum,knum);               
                [aa,bb,cc]=find(GSPC);
                %iind=[iind;aa];jind=[jind;bb];kvalue=[kvalue;cc];
                %MPC(lpindx(i):lpindx(i+1)-1,(k-1)*knum+1:k*knum)=GSPC;%(iper(k,:),iper(k,:));
                %clear GSPC;
            end   
            if(pc_type==3)
                [L,U,P]=luinc(PK,1e-04);
                LUPC=L*U;
                [aa,bb,cc]=find(LUPC);
                %MPC(lpindx(i):lpindx(i+1)-1,(k-1)*knum+1:k*knum)=LUPC;%(iper(k,:),iper(k,:));
            end
            
        else if (pc_type>3 & pc_type<=5)
                
                if(pc_type==4)
                    JPC=1/w.*(diag(diag(PK,0),0)+diag(diag(PK,1),1)+diag(diag(PK,-1),-1));     
                    [aa,bb,cc]=find(JPC);
                    %MPC(lpindx(i):lpindx(i+1)-1,(k-1)*knum+1:k*knum)=JPC;%(iper(k,:),iper(k,:));
                end
                if(pc_type==5)
                    GSPC=sparse(tril(PK,-2))+1/w.*(spdiags(spdiags(PK,-1),-1,knum,knum)+spdiags(spdiags(PK,0),0,knum,knum)+...
                        spdiags(spdiags(PK,1),1,knum,knum)); 
                    [aa,bb,cc]=find(GSPC);
                    %MPC(lpindx(i):lpindx(i+1)-1,(k-1)*knum+1:k*knum)=GSPC;%(iper(k,:),iper(k,:));
                    clear GSPC;
                end
            else
                disp('error: undefined smoother')
                return
            end
        end
        iind=[iind;lpindx(i)+aa-1];jind=[jind;(k-1)*knum+bb];kvalue=[kvalue;cc];
        clear per iper secs this_K this_point PK
    end
end
MPC=sparse(iind,jind,kvalue);
