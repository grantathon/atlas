clear all;
tic;

% Inputs
n = 7;
d = 2;
beta = 0;
v = 1;
% 
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


% Compute Householder parameters for first Q
alpha = -sign(A(2, 1))*norm(A(2:n, 1));
r = sqrt((alpha^2 - A(2, 1)*alpha) / 2);

% Compute householder vector
v = zeros(1, d+1);
v(2) = (A(2, 1) - alpha) / (2*r);
if((d+1) > 2)
    for k=3:(d+1)
        v(k) = A(k, 1) / (2*r);
    end
end     

% Compute Q1
Q1 = eye(d+1) - 2*transpose(v)*v;

% Compute new column of A
A(2:d+1, 1) = Q1(2:d+1, 2:d+1)*A(2:d+1, 1);
A(1, 2:d+1) = A(2:d+1, 1);

% Compute the remaining block-diagonal
A(2:d+1, 2:d+1) = Q1(2:d+1, 2:d+1)*A(2:d+1, 2:d+1)*Q1(2:d+1, 2:d+1);

% Compute the block-diagonal's bottom-right neighbor blocks


A

beta = beta + 1;

% Compute Householder parameters for next Q
alpha = -sign(A(2+beta, 1+beta))*norm(A(2+beta:n, 1+beta));
r = sqrt((alpha^2 - A(2+beta, 1+beta)*alpha) / 2);

% Compute householder vector
% v = zeros(1, dim-1);
% v(2) = (A(2+1, 1+1) - alpha) / (2*r);
% if((dim-1) >= 3)
%     for k=3:(dim-1)
%         v(k) = A(k, 1) / (2*r);
%     end
% end
v = zeros(1, d+1);
v(2) = (A(2+beta, 1+beta) - alpha) / (2*r);
if((d+1) > 2)
    for k=3:(d+1)
        v(k) = A(k, 1) / (2*r);
    end
end

% Compute Q2
% Q2 = eye(dim-1) - 2*transpose(v)*v;
Q2 = eye(d+1) - 2*transpose(v)*v;

% Compute the above block-diagonal's bottom-right neighbors
A(3:n-0, 2) = Q2(2:n-1, 2:n-1)*transpose(A(3:n-0, 2))*Q1(2:n-1, 2:n-1)

A

toc
% clear Q alpha b dim i k r v;