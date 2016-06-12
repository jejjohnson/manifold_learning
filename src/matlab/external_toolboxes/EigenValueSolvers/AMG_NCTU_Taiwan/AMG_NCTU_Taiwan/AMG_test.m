load A_b.mat 
K=A;
F=b;
w=zeros(size(A,1),1);
level=3;
node_order=[-1 1 1];
relax_it=2; 
relax_para=1;  
post_smoothing=1; 
max_iter=200; 
tol=1e-06; 
pc_type=2;
connection_threshold=0.25;
 [w,error,iter,flag]=AMG(K,F, w, level,  relax_it, relax_para, ...
                         post_smoothing, max_iter, tol,  pc_type, connection_threshold);
