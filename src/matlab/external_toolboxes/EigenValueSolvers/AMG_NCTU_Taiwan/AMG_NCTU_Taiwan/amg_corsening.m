function [A_stack,P_stack,m_cindx,l_cindx,lpindx]=AMG_corsening(A,level,theda)

%%% lpindx(i): number of points in level(i)=lpindx(i+1)-lpindx(i)
%%% m_cindx(i,:) : global indedx of points in corser grids
%%% l_cindx(i,:) : pre-level fine grid index of points in current corser grid

[n,n]=size(A);

P_stack=sparse(2*n,n);
A_stack=sparse(n,2*n);
currentA=sparse(n,n);

m_cindx=sparse(level,n);
l_cindx=sparse(level,n);

currentA=A;
%currentpt=pt;

A_stack(1:n,1:n)=A;
P_stack(1:n,1:n)=speye(n);

m_cindx(1,:)=[1:n];
l_cindx(1,:)=[1:n];

lstart=1;
lpindx(1)=1;
lpindx(2)=lpindx(1)+n;

fn=n;
cn=fn;

for i=2:level
   %strong_connect_time=cputime;
   [ni,s,st,lamda]=full_strong_connection(A_stack(1:cn,lstart:lstart+cn-1),theda); 
   %strong_connect_time=cputime-strong_connect_time
   disp('pass strong_connection construction')
   
   %prec_time=cputime;
   %[c,f,u] = new_pre_cpoint(A_stack(1:cn,lstart:lstart+cn-1),s,st,lamda);
   [c,f,u] = pre_cpoint(A_stack(1:cn,lstart:lstart+cn-1),s,st,lamda);
   %prec_time=cputime-prec_time
   disp('pass preliminary c-points selection')
   oldc=c;
   oldf=f;
   %postc_time=cputime;
   [c,f,c_to_f]=post_cpoint(A_stack(1:cn,lstart:lstart+cn-1),s,oldc,oldf,ni);
   %postc_time=cputime-postc_time
   disp('pass final c-points selection ')

   cpindx=[];
   fpindx=[];
   cpindx=find(c(:)>0);
   fpindx=find(f(:)>0);
   lstart=lstart+cn;

   [cn,cm] = size(cpindx);
   [fn,fm]=size(fpindx);
   prolong=sparse(n,cn);
   for j=1:fn,
      prolong(f(1,fpindx(j)),:)=c_to_f(f(fpindx(j)),c(cpindx));
   end
   for j=1:cn,
      prolong(c(1,cpindx(j)),j)=1.0;
   end

   [t1,t2]=size(prolong);
   P_stack(1:n,lstart:lstart+cn-1)=prolong;
   corser_A=sparse(cn,cn);
   corser_A=prolong'*A_stack(1:n,lstart-n:lstart-1)*prolong;
   
   m_cindx(i,1:cn)=m_cindx(i-1,c(cpindx));
   l_cindx(i,1:cn)=c(cpindx);
   lpindx(i+1)=lpindx(i)+cn;

   A_stack(1:cn,lstart:lstart+cn-1)=sparse(cn,cn);
   A_stack(1:cn,lstart:lstart+cn-1)=corser_A;
   n=cn;
   ni=[];s=[];st=[];c=[];f=[];u=[];
   oldc=[];oldf=[];cpt=[];fpt=[];rcpt=[];
   prolong=[];c_to_f=[];lamda=[];
end

