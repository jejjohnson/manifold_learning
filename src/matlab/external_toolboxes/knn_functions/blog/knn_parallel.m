%% P1  and P2 are Nx3 matrices

size(P1)

nel_P1 = size(P1,1);
nc = 8; % number of cores
chunck = floor(nel_P1/nc);

P1_ ={};
for i=0:nc-2
    ii = i*chunck +1;
    ff = (i+1)*chunck;
    P1_{i+1} = P1(ii:ff,:);
end
P1_{nc} = P1(ff+1:end,:);
%%

Mdl = KDTreeSearcher(P2);

IDX_ = {};
DIST_ = {};

parpool(nc);

tic
parfor i=1:nc
[IDX_{i}, DIST_{i}] = knnsearch(Mdl, P1_{i});
end
toc

%%
idxP2n4 = zeros(size(P1,1),1);
dist_kd4 = zeros(size(P1,1),1);

for i=0:nc-2
    ii = i*chunck +1;
    ff = (i+1)*chunck;
    idxP2n4(ii:ff) = IDX_{i+1};
    dist_kd4(ii:ff) = DIST_{i+1};
end
idxP2n4(ff+1:end,:) = IDX_{nc};
dist_kd4(ff+1:end,:) = DIST_{nc};

P2n4 = P2(idxP2n4,:);

% check results


tic
[idxP2n3, dist_kd3] = knnsearch(Mdl,P1(:,:));
toc
P2n3 = P2(idxP2n3,:);

disp(isequal(dist_kd4,dist_kd3));
disp(isequal(P2n4,P2n3));
disp(isequal(idxP2n4(:,1),idxP2n3))
