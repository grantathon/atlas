function [b, r] = BrunoLang_2_1_Find_b_r(n, d, u)
%   [b, r] = BrunoLang_2_1_Find_b_r(n, d, u) - Selects the appropriate
%   iteration parameters for Bruno Lang's symmetric banded matrix
%   tridiagonal reduction 2.1 algorithm.  Returns [0, 0] if there exists
%   no sufficient values for b and r, which indicates the algorithm is
%   finished.
%
%   n - Domain elements
%   d - Semi-bandwidth
%   u - Current reduction step

% Educated guess for parameters
b = floor((n - u + 1)/2);
r = n - u - d*(b - 1);

% Confirm validity of computed parameters & optimize if possible
if(r > b)
    while((r > b) && (r > 0) && (b > 0))
        b = b - 1;
        r = n - u - d*(b - 1);
    end
    
    if((r > b) || (r < 1) || (b < 1))
       b = 0;
       r = 0;
    end
elseif(r < 1)
    b = floor((n - u - 1)/d) + 1;
    r = n - u - d*(b - 1);
    
    if(r > b)
        while((r > b) && (r > 0) && (b > 0))
            b = b - 1;
            r = n - u - d*(b - 1);
        end

        if((r > b) || (r < 1) || (b < 1))
           b = 0;
           r = 0;
        end
    end
end

end