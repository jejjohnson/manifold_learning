function C = CountSketch(A,s)
[m, n] = size(A);
sgn = randi(2, [1, n] ) * 2 - 3;
A = bsxfun(@times, A, sgn);
ll = randsample(s, n, true);
C = zeros(m, s);

for j = 1:n
    C(:, ll(j)) = C(:, ll(j))+A(:,j);
end

end