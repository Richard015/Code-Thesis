% This class generates the block diagonal matrix Sigma_u1 of the
% cross-sectional independence model
function RR = BlockGenThesis(nBlock,BlockSize,rho,n)
% generate block diagonal matrix such that RR'*RR = block diagonal

SigU = eye(n);  % SigmaU

% generate block
SSS = eye(BlockSize,BlockSize);
for i = 1:BlockSize
    for j = i+1:BlockSize
        SSS(i,j) = rho^abs(i-j);
        SSS(j,i) = rho^abs(i-j);
    end
end

% assign block to Sig_U
for iBlock = 1:nBlock
    pos = (BlockSize*(iBlock-1) + 1):(BlockSize*iBlock);
    SigU(pos,pos) = SSS;
end

RR=chol(SigU);

end