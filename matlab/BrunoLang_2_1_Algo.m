clear all;
tic;

% Inputs
n = 10;  % A_nxn (i.e., square A with dim n)
d = 3;  % semi-diagonal size (e.g., 1 = trididagonal A)

% Initialize unit vector
unit = zeros(d, 1);
unit(1) = 1;

% UNCOMMENT THIS CODE BLOCK TO DYNAMICALLY CREATE SYMMETRIC TOEPLITZ
% MATRICES. DON'T FORGET THE COMMENT OUT THE A MATRICES BELOW!
%
% Build symmetric Toeplitz
% temp = rand(1, (d+1));
% for i=(d+2):n
%     temp = [temp 0];
% end
% A = toeplitz(temp)
% 
% clear temp;

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
% 
% A = [   0.84 0.39 0.78 0.00 0.00 0.00 0.00;
%         0.39 0.84 0.39 0.78 0.00 0.00 0.00;
%         0.78 0.39 0.84 0.39 0.78 0.00 0.00;
%         0.00 0.78 0.39 0.84 0.39 0.78 0.00;
%         0.00 0.00 0.78 0.39 0.84 0.39 0.78;
%         0.00 0.00 0.00 0.78 0.39 0.84 0.39;
%         0.00 0.00 0.00 0.00 0.78 0.39 0.84; ]
% 
% A = [
%     0.9649    0.1576    0.9706         0         0         0         0         0         0         0;
%     0.1576    0.9649    0.1576    0.9706         0         0         0         0         0         0;
%     0.9706    0.1576    0.9649    0.1576    0.9706         0         0         0         0         0;
%          0    0.9706    0.1576    0.9649    0.1576    0.9706         0         0         0         0;
%          0         0    0.9706    0.1576    0.9649    0.1576    0.9706         0         0         0;
%          0         0         0    0.9706    0.1576    0.9649    0.1576    0.9706         0         0;
%          0         0         0         0    0.9706    0.1576    0.9649    0.1576    0.9706         0;
%          0         0         0         0         0    0.9706    0.1576    0.9649    0.1576    0.9706;
%          0         0         0         0         0         0    0.9706    0.1576    0.9649    0.1576;
%          0         0         0         0         0         0         0    0.9706    0.1576    0.9649;
% ]

A = [
    0.9572    0.4854    0.8003    0.1419         0         0         0         0         0         0;
    0.4854    0.9572    0.4854    0.8003    0.1419         0         0         0         0         0;
    0.8003    0.4854    0.9572    0.4854    0.8003    0.1419         0         0         0         0;
    0.1419    0.8003    0.4854    0.9572    0.4854    0.8003    0.1419         0         0         0;
         0    0.1419    0.8003    0.4854    0.9572    0.4854    0.8003    0.1419         0         0;
         0         0    0.1419    0.8003    0.4854    0.9572    0.4854    0.8003    0.1419         0;
         0         0         0    0.1419    0.8003    0.4854    0.9572    0.4854    0.8003    0.1419;
         0         0         0         0    0.1419    0.8003    0.4854    0.9572    0.4854    0.8003;
         0         0         0         0         0    0.1419    0.8003    0.4854    0.9572    0.4854;
         0         0         0         0         0         0    0.1419    0.8003    0.4854    0.9572;  
]

% Main loop
for u = 1:(n-2)
    % Determine b & r iteration parameters
    b = floor((n - u) / d);
    r = n - u - d*(b - 1);
    if(r > d)
        b = b + floor(r / d);
        r = r - d*floor(r / d);
    end

    if(b == 1)
        d = r;
    end
    
    Q1 = ComputeQ(A((u+1):n, u), d);
    
    % Compute new column/row of A
    A((u+1):(u+d), u) = Q1 * A((u+1):(u+d), u);
    A(u, (u+1):(u+d)) = A((u+1):(u+d), u);
    
    % Distribute the Qs among the other parts of A for proper trasformation
    for beta = 1:b
        % Compute the block-diagonal
        if(beta ~= b)
            bd_idx_a = 2 + u - 1 + d*(beta - 1);
            bd_idx_b = 2 + u - 1 + d*(beta - 1) + d - 1;
            
            A(bd_idx_a:bd_idx_b, bd_idx_a:bd_idx_b) = transpose(Q1) * A(bd_idx_a:bd_idx_b, bd_idx_a:bd_idx_b) * Q1;
        else
            if(b == 1)
                bd_idx_c = n - d + 1;
                bd_idx_d = n;
            end
            
            A(bd_idx_c:bd_idx_d, bd_idx_c:bd_idx_d) = transpose(Q1) * A(bd_idx_c:bd_idx_d, bd_idx_c:bd_idx_d) * Q1;
        end
            
        % Compute the block-diagonal's bottom-right neighbor blocks
        if(beta < b-1)
            bd_idx_c = bd_idx_a + d;
            bd_idx_d = bd_idx_b + d;
            
            % Compute Q2 for upper-/lower-triangularization of neighbors
            A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) = A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) * Q1;
            w = A(bd_idx_c:bd_idx_d, bd_idx_a) + sign(A(bd_idx_c, bd_idx_a))*norm(A(bd_idx_c:bd_idx_d, bd_idx_a)).*unit;
            v = w./norm(w);
            Q2 = eye(d) - 2*v(1:d)*transpose(v(1:d));
            
            % Compute new bottom/right neighbors of block-diagonal of A
            A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) = transpose(Q2) * A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b);
            A(bd_idx_a:bd_idx_b, bd_idx_c:bd_idx_d) = transpose(A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b));
            
            % Copy over new Q1
            Q1 = Q2;
        elseif(beta == b-1)
            bd_idx_c = bd_idx_a + d;
            bd_idx_d = bd_idx_b + r;
            
            % Compute Q2 for upper-/lower-triangularization of neighbors
            A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) = A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) * Q1;
            
            % Alter unit vector dimension to r
            unit = zeros(r, 1);
            unit(1) = 1;

            w = A(bd_idx_c:bd_idx_d, bd_idx_a) + sign(A(bd_idx_c, bd_idx_a))*norm(A(bd_idx_c:bd_idx_d, bd_idx_a)).*unit;
            v = w./norm(w);
            Q2 = eye(r) - 2*v(1:r)*transpose(v(1:r));
            
            % Compute new bottom/right neighbors of block-diagonal of A
            A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b) = transpose(Q2) * A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b);
            A(bd_idx_a:bd_idx_b, bd_idx_c:bd_idx_d) = transpose(A(bd_idx_c:bd_idx_d, bd_idx_a:bd_idx_b));

            Q1 = Q2;

            % Reset altered unit vector dimension to d
            unit = zeros(d, 1);
            unit(1) = 1;
        end
    end
end

A

toc
clear unit Q1 Q2 alpha b dim i k r v bd_idx_a bd_idx_b bd_idx_c bd_idx_d beta d n rad u w ans;