function Q = ComputeQ(A, d)
    % Compute Householder parameters
    alpha = -sign(A(1))*norm(A(1:length(A)));
    rad = sqrt((alpha^2 - A(1)*alpha) / 2);
    
    % Compute Householder vector
    v = zeros(d+1, 1);
    v(2) = (A(1) - alpha) / (2*rad);
    for k=3:(d+1)
        v(k) = A(k - 1) / (2*rad);
    end
    
    % Compute Q1
    Q = eye(d) - 2*v(2:(d+1))*transpose(v(2:(d+1))); 
end