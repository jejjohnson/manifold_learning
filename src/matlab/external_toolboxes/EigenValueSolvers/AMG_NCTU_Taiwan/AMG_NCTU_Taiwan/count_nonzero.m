function [n]=count_nonzero(vec)

%[row,col]=size(vec);
%n=0;
[vi,vj]=find(vec~=0);
n=size(vj,2);
%for i=1:col,
%   if(vec(1,i)~=0) n=n+1;
%end
