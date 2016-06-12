function [n,s,st,lamda]=full_strong_connection(A,theda)

[row,col]=size(A);
s=sparse(row,col);
st=sparse(col,row);
n=sparse(row,col);
lamda=sparse(1,col);
A_plus=abs(A);
for i=1:row,
   A_plus(i,i)=0;
   [max_value(i),max_ind(i)]=max(A_plus(i,:));
end
for i=1:row,
   [col]=find(A(i,:));
   csize=size(col);
   cnum=csize(2);
   k=1;l=1;nk=1;
   for c=1:cnum,
      j=col(c);
      if(A_plus(i,j)>theda.*max_value(i)&j~=i)
         s(i,k)=j;
         k=k+1;
      end
      if(A_plus(i,j)~=0)
         n(i,nk)=j;
         nk=nk+1;
      end      
   end
   col=[];
end
col=row;
for i=1:col,
   [row]=find(A(:,i));
   rsize=size(row);
   rnum=rsize(1);
   k=1;l=1;
   for r=1:rnum,
      j=row(r);
      if(A_plus(j,i)>=theda.*max_value(j)&j~=i)
         st(i,l)=j;
         l=l+1;
         lamda(1,i)=lamda(1,i)+1;
      end
   end
   row=[];
end
      
