%This class runs the simulations of the factor model as described in the
%paper
function[p_Wald,p_WaldPE,p_WaldPE2,p_Fan,p_FanPE,p_FanPE2,p_S,p_S2] = sim_scriptie_factor(N,T)
% Simulation -- error covariance is non-diagonal
% ------------------------------------------
% 1) Parameter setup
NN = 2000;               % repetitions
SigLevel = 0.95;        % significance level
h0 = true;      %true if testing null hypothesis, false if testing one of the alternatives
ha = 0;         %1 if testing Ha^1 and 0 if testing Ha^2

%  H0 vs H1
TrueTheta = zeros(N,1);  % H_0
if (ha==1)
    r = max(round(N/T),1);  % H_alpha1
    r_max = round(1.2*r);   % Number of elements in theta_S
    theta=0.3;
else
    r = round(N^0.4); % H_alpha2
    r_max = round(N^0.5); % Number of elements in theta_S
    theta=sqrt(log(N)/(T));
end
if (h0==true)
    theta=0;
end


TrueTheta(1:r) = theta*ones(r,1);
% ----

% threshold for power enhancement
delta_NT =  sqrt(log(N))*log(log(T));
delta_rT =  sqrt(log(r))*log(log(T));
delta_rNT =  sqrt(log(0.25*N+0.75*r))*log(log(T));

% recording
rec_Wald = zeros(NN,1);
rec_WaldPE = zeros(NN,1);
rec_WaldPE2 = zeros(NN,1);
rec_SS = zeros(NN,1);    % record whether screening set S0 is empty
rec_SS2 = zeros(NN,1);   % record whether screening set SN is empty
rec_Fan = zeros(NN,1);
rec_FanPE = zeros(NN,1);
rec_FanPE2 = zeros(NN,1);

for Rep = 1:NN
    
    % generate factor pricing model
    [FF, BB, UU, Sig0]=genFPM_scriptie(N,T);
    
    %generating y_t
    YY = repmat(TrueTheta,1,T) + BB'*FF + UU;
    
    %%%%%  Estimation   %%%%
    f_bar = mean(FF, 2);
    w = ((FF*FF')/T)^(-1) *f_bar;
    a_ft = 1-(f_bar)'*w;
    theta_hat = zeros(N,1);
    
    for t=1:T
        theta_hat = theta_hat + (1-FF(:,t)'*w)*YY(:, t);
    end
    theta_hat = theta_hat/(T*a_ft);
    beta_hat = (YY-theta_hat)*FF'*(FF*FF')^(-1);
    u_hat=YY- beta_hat*FF;
    sum_forvhat = zeros(N, 1);
    for t=1:T
        sum_forvhat = sum_forvhat+u_hat(:, t).^2;
    end
    sum_forvhat = sum_forvhat/(T*a_ft);
    v_hat = sum_forvhat/T;
    %%%%% Construct test statistics %%%%
    
    % PE component proposed by Fan (2015)
    J_0=0;
    for i = 1:N
        if abs(theta_hat(i))> delta_NT*v_hat(i)^(0.5) %screening
            J_0 = J_0 + theta_hat(i)^2 / v_hat(i);
        end
    end
    J_0=sqrt(N)*J_0;
    
    % My proposed power enhancement component J_N
    J_N=0;
    for i = 1:r_max
        if abs(theta_hat(i))> delta_rNT*v_hat(i)^(0.5) %change delta_rNT to delta_rT or delta_NT when testin Jn for different delta
            J_N = J_N + theta_hat(i)^2 / v_hat(i);
         end
    end
    J_N=sqrt(N)*J_N;
    
    % Fan(1996)
    Fan_a = 1/(log(N)^2);
    Fan_delta = sqrt(2*log(N*Fan_a));
    Fan_mu = sqrt(2/pi)*Fan_a^(-1)*Fan_delta*(1+Fan_delta^(-2));
    Fan_sigma = sqrt( sqrt(2/pi)* Fan_a^(-1)*Fan_delta^3*(1+3*Fan_delta^(-2)));
    FanH = 0;
    for i = 1:N
        if abs(theta_hat(i))> Fan_delta*v_hat(i)^(0.5)
            FanH = FanH + theta_hat(i)^2 *v_hat(i)^-1;
        end
    end
    J_thr = (FanH - Fan_mu)/Fan_sigma;
    
    % Wald
    Sigma_u_hat = u_hat*u_hat'/(T-4);
    Constant = log(N)/T;
    for i= 1:N
        for j = i+1:N
            if abs(Sigma_u_hat(i, j)) <= 3.8*sqrt(Sigma_u_hat(i, i) * Sigma_u_hat(j, j)* Constant)
                Sigma_u_hat(i, j) = 0;
                Sigma_u_hat(j, i) = 0;
            end
        end
    end
    Wald = (T*a_ft*theta_hat'*Sigma_u_hat^(-1)*theta_hat-N)/sqrt(2*N);
       
    %%%%%   Recording   %%%%    
    rec_Wald(Rep) = (Wald > norminv(SigLevel,0,1));
    rec_WaldPE(Rep) = (Wald+J_0 > norminv(SigLevel,0,1));
    rec_WaldPE2(Rep) = (Wald+J_N > norminv(SigLevel,0,1));
    rec_SS(Rep) = (J_0==0);
    rec_SS2(Rep) = (J_N==0);
    rec_Fan(Rep) = (J_thr > norminv(SigLevel,0,1));
    rec_FanPE(Rep) = (J_thr+J_0 > norminv(SigLevel,0,1));
    rec_FanPE2(Rep) = (J_thr+J_N > norminv(SigLevel,0,1));
end

p_Wald = sum(rec_Wald)/NN;
p_WaldPE = sum(rec_WaldPE)/NN;
p_WaldPE2 = sum(rec_WaldPE2)/NN;
p_S = sum(rec_SS)/NN;
p_S2 = sum(rec_SS2)/NN;
p_Fan = sum(rec_Fan)/NN;
p_FanPE = sum(rec_FanPE)/NN;
p_FanPE2 = sum(rec_FanPE2)/NN;

% printing result
disp('------------  Non-diagonal error covariance  ----------------')
fprintf('N: %s, T: %s, alpha0: %s, replication: %s\n',num2str(N),num2str(T),num2str(theta),num2str(NN));
fprintf('p-values: Wald     WaldPE     WaldPE2     Fan     FanPE       FanPE2     S\n      S2\n');
fprintf('    %10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f\n',p_Wald,p_WaldPE,p_WaldPE2,p_Fan,p_FanPE,p_FanPE2,p_S,p_S2);
end



