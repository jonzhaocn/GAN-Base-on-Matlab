%if x>=0 x
%if x<0 leak*x
function result = leaky_relu(x, leak)
    if (nargin < 2)
        leak = 0.2;
    end
    result = x;
    result(x<0) = result(x<0) * leak;
end