function [dist, idx] = knn_parallel(data, num_cores)



size(data)

nel_P1 = size(data,1);
nc = num_cores; % number of cores
chunck = floor(nel_P1/nc);

P1_ ={};
for i=0:nc-2
    ii = i*chunck +1;
    ff = (i+1)*chunck;
    P1_{i+1} = data(ii:ff,:);
end
P1_{nc} = data(ff+1:end,:);
%%

Mdl = KDTreeSearcher(data);

IDX_ = {};
DIST_ = {};

parpool(nc);


parfor i=1:nc
[IDX_{i}, DIST_{i}] = knnsearch(Mdl, P1_{i});
end



%%
idxP2n4 = zeros(size(data,1),1);
dist_kd4 = zeros(size(data,1),1);

for i=0:nc-2
    ii = i*chunck +1;
    ff = (i+1)*chunck;
    idxP2n4(ii:ff) = IDX_{i+1};
    dist_kd4(ii:ff) = DIST_{i+1};
end
idxP2n4(ff+1:end,:) = IDX_{nc};
dist_kd4(ff+1:end,:) = DIST_{nc};

P2n4 = data(idxP2n4,:);

idx = P2n4;
dist = P2n4;

delete(gcp('nocreate'))

end