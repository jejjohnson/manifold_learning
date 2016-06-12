function [per,iper]=V_renumber(opt,btflag,lrflag)

ptsize=size(opt);
spt(:,1)=lrflag.*opt(1,:)';
spt(:,2)=btflag.*opt(2,:)';
[rtemp,per]=sortrows(spt,[1,2]);
for i=1:ptsize(2),
   iper(per(i))=i;
end
