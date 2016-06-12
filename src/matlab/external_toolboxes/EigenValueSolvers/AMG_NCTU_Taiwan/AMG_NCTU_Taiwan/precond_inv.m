function [w,r]=precond_inv(f,M,A,w,pernum,relax_iter,n)

for i=1:relax_iter,
    for piter=1:pernum,
        Ml=M(1:n,(piter-1)*n+1:piter*n);
        c = Ml\(f-(A*w));
        w=w+c;
    end
end
r=f-A*w;
