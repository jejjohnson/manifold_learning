function [mw, err, iter, flag,clevel]=MG_solver(A_stack, MPC, mw, b_int, relax_it, relax_para, post_smoothing,...
          tol, iter, max_iter,  max_level, cycle, P_stack, m_cindx, lindx)

%%%
% relax_it: smoothing steps for Multigrid
% relax_para: relaxation parameter used in SOR MG smoother
% max_iter: Max allowed MG iterations
% tol: residual tolerance for MG to stop
% tol_type: 1: discrete l2-norm, 2:contineous l2-norm
% iter: MG iteration counts
% max_level: Max grid level on each cycle
% other parameters : see MG.m discription

%%% mw = initial guess %%%%                                              
iter=0;
[pnum,temp]=size(mw);
itsize=size(tol,2);
err(1)=0;
if(max_iter>1)
    resd=b_int-A_stack(1:pnum,1:pnum)*mw;
    err(1)=norm(resd)/norm(b_int);
else
    err(1)=1;   %%% for 1 step v-cycle multigrid preconditioner in GMRES 
end

ini_iteration=1;
clevel=max_level;
pre_err=0;
flag=0;
true_x=sparse(pnum,max_level+1);
while (ini_iteration==1|(err(iter+1)>tol&iter<max_iter)),
    ini_iteration=0;
    pre_err=err(iter+1);
    
    [mw, resd, iter, flag, clevel] = MG(A_stack,MPC, mw, b_int, relax_it, relax_para, post_smoothing,...
        tol, iter,  clevel, max_level, cycle, P_stack, m_cindx, lindx);
    
    err(iter+1)=norm(resd)/norm(b_int);                    
    if(err(iter+1)<tol)
        flag=1;
    end         
    conv_rate=err(iter+1)/pre_err;    
end
