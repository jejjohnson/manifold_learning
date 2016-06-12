function P = completeGraph(ind)
% construct all combinations of indices chosen two at a time
n = numel(ind);

i = repmat(ind(:)',[n 1]);
j = repmat(ind(:),[1 n]);

T = triu(true(n),1);

P = [i(T) j(T)];
