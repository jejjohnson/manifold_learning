function G = similarityGraph(Y, label)

% find all the indices equal to label
tic;
ind = find(Y==label);
toc;




tic;
Y(idx(:), idy(:)) = 1;
toc;

% % create matrix with indices that are equal to
% Gs = sparse(repmat(Y,1,length(Y)));
% Gst = sparse(repmat(Y,1,length(Y)))';
% 
% % find where these matrices are the same
% G = Gs == Gst;
% 
% % wherever Y was zero, should still be zero
% G(Y==0, :) = 0;
% G(:, Y==0) = 0;
% 
% % make double precision
% G = double(G);






end