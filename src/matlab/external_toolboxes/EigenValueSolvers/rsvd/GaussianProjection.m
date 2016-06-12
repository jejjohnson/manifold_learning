function C = GaussianProjection(A,s)

n = size(A, 2);
S = randn(n, s) / sqrt(s);
C = A * S;

end