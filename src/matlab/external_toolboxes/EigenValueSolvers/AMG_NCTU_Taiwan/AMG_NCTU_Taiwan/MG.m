function [x, resd, iter, flag, clevel] = MG(As,Mls, x, b, relax_iter,relax_para,post_smoothing,...
    tol,iter, clevel,max_level,cycle,P_stack,m_cindx,lindx)

% m_cindx point to global index for nodes in each level %
% lindx record the start indx and end indx of each level for extracting 
% matrix A, Prolongation operator and Restriction operator from As,P_stack,R_stack
% m_cindx and Mrs are no use at this stage %%%
% relax_iter: smoothing steps
% cycle: multigrid cycle types
% iter: multigrid iteration counts
% clevel: current grid level on calling MG

pernum= 1;
head=lindx(max_level-clevel+1);
tail=lindx(max_level-clevel+2)-1;
delta_head_tail=tail-head+1;
A=As(1:delta_head_tail,head:tail);
[n,n] = size(A);          
delta_x=zeros(n,1); 

if(max_level==0)
    
    Ml=Mls(head:tail,1:pernum*delta_head_tail);    
    prolong=P_stack(head:tail,head:tail);
    cx=zeros(n,1);
    prelax=0;
    if(size(relax_iter,2)==1)
        [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter*(2^prelax),n);
    else
        [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter(1),n);
    end        
    iter=iter+1;
    clevel=max_level;
    return
end

if(clevel>0)
    
    Ml=Mls(head:tail,1:pernum*delta_head_tail);
    prelax=0;
    clevel=clevel-1; 
    chead=lindx(max_level-clevel+1);
    ctail=lindx(max_level-clevel+2)-1;
    cdelta_head_tail=lindx(max_level-clevel+2)-lindx(max_level-clevel+1);
    prolong=P_stack(1:delta_head_tail,chead:ctail);
    restrict=prolong';
    resd=b-A*x;
    
    if(size(relax_iter,2)==1)
        [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter-(max_level-clevel)+1,n);
    else
        [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter(max_level-clevel),n);
    end        
                    
    corser_b=restrict*resd;
    cini=zeros(cdelta_head_tail,1);
    [cx, resd, iter, flag, clevel]=MG(As, Mls, cini, corser_b, ...
        relax_iter,relax_para,post_smoothing,tol,iter,clevel,max_level,cycle,P_stack,m_cindx,lindx);
    
end

if(clevel==0&max_level>0)
    x=A\b;
    clevel=clevel-1;
    resd=zeros(n,1);
    iter=iter;
    flag=0;
    return
end

if(clevel<0&clevel+max_level>0)
    x=x+prolong*cx;
    if(post_smoothing==1)
        if(size(relax_iter,2)==1)
            [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter*(2^prelax),n);
        else
            [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter(max_level+clevel+1),n);
        end        
    else
        resd=b-A*x;                  
    end
    iter=iter;
    flag=0;
    clevel=clevel-1;
    return
end

if((clevel<0&clevel+max_level==0)|max_level==0)
    x=x+prolong*cx;    
    resd=b-A*x;    
    if(post_smoothing==1)
        if(size(relax_iter,2)==1)
            [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter*(2^prelax),n);
        else
            [x,resd]=precond_inv(b,Ml,A,x,pernum,relax_iter(max_level+clevel+1),n);
        end        
    else
        resd=b-A*x;    
    end
    iter=iter+1;
    clevel=max_level;    
end



