function [per,iper,secs,actual_block_num,flag]=reordering(pt,preorder)

actual_block_num=1;
psize=size(pt);
knum=psize(2);
per=[1:knum]';
iper=[1:knum];
flag=1;
secs=knum;
if(preorder(1) >=0)
    switch preorder(1)
    case 0
        [per,iper]=h_renumber(pt,preorder(2),preorder(3));
    case 1
        [per,iper]=v_renumber(pt,preorder(2),preorder(3));
    end
end
   
