%This class runs the simulations of the cross-sectional independence model
%as described in the paper
function[p_Quad,p_QuadPE,p_QuadPE2,p_S,p_S2] = sim_scriptie_csi(n,T)
% Testing cross-sectional independence

%-------------------------------------
% 1) Set up parameters
rep = 2000;           % replications 2000
N = n*(n-1)/2;        % number of parameters
nBlock = floor(n^.3);  % number of blocks in the covariance
BlockSize = 4;
rho = 0.2;
r = 1.5*n;          %Number of off-diagonal entries per block-diagonal matrix is 6. There are n/4 block diagonal matrices. So r=|theta_S|=6*n/4=1.5n

Hypothesis = 0;     % Hypothesis is 0 when testing H0 and 1 when testin Ha

delta_NT = 2.25*log(N)*(log(log(T)))^2;
delta_rT = 2.25*log(r)*(log(log(T)))^2;
delta_rNT = 2.25*log((N+3*r)/4)*(log(log(T)))^2;
%tt = rate/T;

% recording
QuadTest = zeros(rep,1);
QuadPETest = zeros(rep,1);
QuadPETest2 = zeros(rep,1);
SS = zeros(rep,1); % record whether screening set S0 is empty 
SS2 = zeros(rep,1); % record whether screening set Sn is empty
AvgMaxSignal = zeros(rep,1);

%----------------------------------------
% 2) Set up hypothesis
if(Hypothesis==1)
    RR = BlockGenThesis(nBlock,BlockSize,rho,n);  % Constructing Block Diagonal
else
    RR = eye(n);
end

%-----------------------------------------
% 3) Testing
for irep = 1:rep
    
    % generate the random effect and the idiosyncratic error
    mu1 = randn(n,1)*sqrt(0.25); %random effect
    mu = mu1*ones(1,T); %idiosyncratic error
    
    % set up X as AR processes
    X = zeros(n,T);
    X(:,1)=0.5*ones(n,1);
    for ix = 2:T
        X(:,ix) = X(:,ix-1)*0.7 + mu1 + randn(n,1);
    end
    
    % heteroskedastic errors
    IndvVar = (0.5*mean(X,2)+1).^2;
    ScaleVar = sqrt(IndvVar/mean(IndvVar));
    e = RR'*diag(ScaleVar)*randn(n,T);
    
    % generate Y
    Y = -ones(n,T)+2*X+mu+e;
    
    %%%%  estimation  %%%%
    tildeY = Y-mean(Y,2)*ones(1,T);
    tildeX = X-mean(X,2)*ones(1,T);
    B = reshape(tildeX,n*T,1)'*reshape(tildeY,n*T,1);
    A = reshape(tildeX,n*T,1)'*reshape(tildeX,n*T,1);
    beta = B/A;
    tildeU = tildeY - tildeX*beta;  % n by T
    Su = tildeU*tildeU'/(T-1);
    R = corrcov(Su);
    
    
    %%%% constructing test  %%%%
    
    % 1. quadratic test J1
    sum1 = (norm(R,'fro')^2-n)/2;
    J1 = (sum1*T-N)/sqrt(n*(n-1))-n/(2*(T-1));
    QuadTest(irep) = J1>norminv(0.95);
    
    % 2. power enhancement J0
    off = R-eye(n);
    num = off.^2;
    denom = (ones(n,n)-num).^2;
    A1 = num./denom;  % rho^2/ (1-rho^2)^2
    AvgMaxSignal(irep) = max(max(A1));
    A2 = A1.*(A1>delta_N/T);  % rho^2/ (1-rho^2)^2 *  1{ rho^2/ (1-rho^2)^2>delta^2/T}
    J0 = sum(sum(A2))/2*T*sqrt(N);
    QuadPETest(irep) = J0+J1>norminv(0.95);
    
    % 3. power enhancement Jn
    A3= zeros(n,n);
    for iBlock = 1:n/4
        for i=4*(iBlock-1) + 1:4*iBlock
            for j=i+1:4*iBlock
                A3(i,j) = A1(i,j)*(A1(i,j)>delta_rNT/T); % Change delta_rNT to delta_rT or delta_NT when testing Jn for different delta
            end
        end
    end
    Jn = sum(sum(A3))*T*sqrt(N);
    QuadPETest2(irep) = Jn+J1>norminv(0.95);
    
    % 4. Screening set S0
    SS(irep) = (J0==0);
    
    % 5. Screening set Sn
    SS2(irep) = (Jn==0);
    
end;

p_Quad = sum(QuadTest)/rep;
p_QuadPE = sum(QuadPETest)/rep;
p_QuadPE2 = sum(QuadPETest2)/rep;
p_S = sum(SS)/rep;
p_S2 = sum(SS2)/rep;

% printing result
fprintf('n: %s  ,T: %s,  H: %s\n',num2str(n),num2str(T),num2str(Hypothesis));
fprintf('p-values:  Quad       Quad+J0      Quad+Jn      SS\n        SS2\n')
fprintf('       %10.4f%10.4f%10.4f%10.4f%10.4f\n',p_Quad,p_QuadPE,p_QuadPE2,p_S,p_S2);
end




