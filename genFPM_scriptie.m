
function [FF, BB, UU, Sigma_u, c] = genFPM_scriptie(N,T)
    % factor pricing model
    % non-Diagonal case
 
    
    % 1) generating factors
    %-------------------
    
    Mu_f=[0.026;0.0211;-0.0043];
    Sigma_f=[3.2351,0.1783,0.7783;0.1783,0.5069,0.0102;0.7783,0.0102,0.6586];
    R_f=chol(Sigma_f);

    FF = repmat(Mu_f,1,T) + R_f'*randn(3,T);
    
    % 2) factor loadings
    %---------------------
    Mu_b=[0.9833,-0.1233,0.0839]';
    Sigma_b=[0.0921,-0.0178,0.0436;-0.0178,0.0862,-0.0211;0.0436,-0.0211,0.7624];
    R_b=chol(Sigma_b);

    BB = repmat(Mu_b,1,N) + R_b'*randn(3,N);
    
    
    % 3) generating covariance matrix
    %--------------------------------
    
    % block diagonal matrix
    Sigma_u = zeros(N,N);
    N1 = 4;
 for iBlock=1:N/N1   % N1 is size of block
     c = unifrnd(0,0.5);
        % set up block 
        SubBlock = zeros(N1,N1);
        for i=1:N1
            for j=1:N1
                if i==j
                    SubBlock(i,j)=1;
                else
                    SubBlock(i,j)=c;
                end
            end
        end                    

        tmp_range = (iBlock-1)*N1+1:(iBlock-1)*N1+4;
        Sigma_u(tmp_range,tmp_range) = SubBlock;
        
  end
      
    
    R_u = chol(Sigma_u);
    UU = R_u'*randn(N,T);

end