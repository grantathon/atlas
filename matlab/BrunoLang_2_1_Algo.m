clear all;
tic;

% Inputs
n = 7;  % A_nxn (i.e., square A with dim n)
d = 2;  % semi-diagonal size (e.g., 1 = trididagonal A)

% Initialize unit vector
unit = zeros(d, 1);
unit(1) = 1;

% Build symmetric Toeplitz
% temp = rand(1, (diagRadius+1));
% for i=(diagRadius + 2):dim
%     temp = [temp 0];
% end
% A = toeplitz(temp);
% 
% clear temp diagRadius;

% A = [ 4 1 -2 2;
%       1 2 0 1;
%       -2 0 3 -2;
%       2 1 -2 -1; ];
%   
% A = [   0.78 0.39 0.84 0.00 0.00;
%         0.39 0.78 0.39 0.84 0.00;
%         0.84 0.39 0.78 0.39 0.84;
%         0.00 0.84 0.39 0.78 0.39;
%         0.00 0.00 0.84 0.39 0.78; ]

A = [   0.78 0.39 0.84 0.00 0.00 0.00 0.00;
        0.39 0.78 0.39 0.84 0.00 0.00 0.00;
        0.84 0.39 0.78 0.39 0.84 0.00 0.00;
        0.00 0.84 0.39 0.78 0.39 0.84 0.00;
        0.00 0.00 0.84 0.39 0.78 0.39 0.84;
        0.00 0.00 0.00 0.84 0.39 0.78 0.39;
        0.00 0.00 0.00 0.00 0.84 0.39 0.78; ]

% Main loop
for u = 1:(n-2)
    % Determine b & r iteration parameters
    [b, r] = BrunoLang_2_1_Find_b_r(n, d, u);
    
    % TODO: Find out what we should do when b & r are no longer computable
    if(b == 0)
        disp('The parameters b & r are undeterminable. Algo terminated!')
        break;
    end

    % Compute Householder parameters
    alpha = -sign(A((u+1), u))*norm(A((u+1):(u+d), u));
    rad = sqrt((alpha^2 - A((u+1), u)*alpha) / 2);

    % Compute Householder vector
    v = zeros(d+1, 1);
    v(2) = (A((u+1), u) - alpha) / (2*rad);
    for k=3:(d+1)
        v(k) = A((k + (u-1)), u) / (2*rad);
    end
    
%     v = zeros(d, 1);
%     v(1) = (A((u+1), u) - alpha) / (2*rad);
%     for k=2:d
%         v(k) = A(k, u) / (2*rad);
%     end
    
    % Compute Q1
    Q1 = eye(d) - 2*v(2:(d+1))*transpose(v(2:(d+1)));
%     Q1 = eye(d) - 2*v*transpose(v);

    % Compute new column/row of A
    A((u+1):(u+d), u) = Q1 * A((u+1):(u+d), u);
    A(u, (u+1):(u+d)) = A((u+1):(u+d), u);
    
    % Distribute the Qs among the other parts of A for proper trasformation
    for beta = 1:b
        bd_idx_a = u + 2*beta - 1;
        bd_idx_b = u + beta*d;
        
        % Compute the block-diagonal
        A(bd_idx_a:bd_idx_b, bd_idx_a:bd_idx_b) = transpose(Q1) * A(bd_idx_a:bd_idx_b, bd_idx_a:bd_idx_b) * Q1;
        
        % Compute the block-diagonal's bottom-right neighbor blocks
        if(beta < b)
            bd_idx_c = bd_idx_a + d;
            bd_idx_d = bd_idx_b + d;
            
            % Compute Q2 for upper-/lower-triangularization of neighbors
            A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) = A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) * Q1;
            w = A(bd_idx_c:(bd_idx_d), bd_idx_a) + sign(A(bd_idx_c, bd_idx_a))*norm(A(bd_idx_c:(bd_idx_d), bd_idx_a)).*unit;
            v = w./norm(w);
            Q2 = eye(d) - 2*v(1:d)*transpose(v(1:d));
            
            % Compute new bottom/right neighbors of block-diagonal of A
            A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) = transpose(Q2) * A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b);
            A(bd_idx_a:bd_idx_b, bd_idx_c:bd_idx_d) = transpose(A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b));
            
            % Copy over new Q1
            Q1 = Q2;
        end
    end
end

A

toc
clear unit Q1 Q2 alpha b dim i k r v bd_idx_a bd_idx_b bd_idx_c bd_idx_d beta d n rad u w ans;