function [w,error,iter,flag]=AMG(K,F, w, level,  relax_it, relax_para, ...
                                 post_smoothing, max_iter, tol,  pc_type, connection_threshold)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Algebraic multigrid code 2012/03/22                                                                                                                                             %%   
%% Author: Dr. Chin-Tien Wu                                                                                                                                                                    %%
%%                Department of Applied Mathematics,                                                                                                                              %%
%%                National Chiao-Tung University, Shinchu, Taiwan                                                                                                        %%
%%                                                                                                                                                                                                                     %%
%% The AMG algorithm implemented here is originated by Ruge and Stuben in 86 and can also be found                     %%
%% in the course note by C. Wagner (http://www.iwr.uni-heidelberg.de/groups/techsim/chris/amg.pdf)                       %%
%% If any error is found, please contact the author at ctw@math.nctu.edu.tw                                                                          %%
%% Disclaimer:                                                                                                                                                                                                %%
%% This program is distributed in the hope that it will be useful, but                                                                                             %%
%% WITHOUT ANY WARRANTY; without even the implied warranty of                                                                                      %%
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                                                                            %%
%%                                                                                                                                                                                                                     %%
%% If you use this AMG code in any program or publication, please acknowledge its author. Thank you very much.   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%     input parameters   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K: input matrix
% F: input righthand side
% w: initial guess
% relax_it: number of smoothing steps
% relax_para: relaxation parameter (1: usual gauss-seidel and jacobi)
% post_smoothing: 1: enable, 0: disable postsmoothing step 
% max_iter: maximal AMG iteration steps
%tol: stopping tolerance on resdiual
% pc_type: 1: jacobi, 2: gauss-seidel, 3: ilu, 4: line jacobi, 5: line gauss-seidel
% connection_threshold: threshold value for building strong connection. Set to 0.25 typically
%%%%%%%%%%%%%%%%%%%%%%%    input parameters      %%%%%%%% %%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%    output parameters    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% w : solution                                                                                      
% error : vector of the ratio of residual and the initial righthand side along iterations           %
% iter: number of iteration steps for AMG to converge to a given tol
% flag: 1: converge, 0: not converge
%%%%%%%%%%%%%%%%%%%%%%%    output parameters     %%%%%%%% %%%%%%%%%%%%%%%%%%%%%% 


max_level=level-1;
clevel=max_level;

A_int=K;b_int=F;
corsening_time=cputime;
disp('time for coarsening')
[A_stack,P_stack,m_cindx,l_cindx,lpindx]=AMG_corsening(A_int,level,connection_threshold);
corsening_time=cputime-corsening_time
MPC_time=cputime;
[MPC]=multi_pcond(A_stack,lpindx,m_cindx,l_cindx,level,relax_para,pc_type);
MPC_time=cputime-MPC_time;
Mrs=[];
iter=0;
MG_time=cputime;
disp('time for MG iteratoin')
[w, error, iter, flag,clevel]=MG_solver(A_stack, MPC, w, b_int, relax_it, relax_para,post_smoothing,...
    tol,iter,max_iter,max_level,1,P_stack,m_cindx,lpindx);

MG_time=cputime-MG_time
if(flag==1)
    s=strcat('AMG converges in  ', num2str(iter));
    disp([s, ' steps.'])
else
    disp('AMG does not converges')
end
clear A_int b_int pt_int A_stack P_stack MPC m_cindx mw

