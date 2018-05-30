function result = delta_leaky_relu(x, leak)
    if (nargin < 2)
        leak = 0.2;
    end
    result = x;
    result(x>=0) = 1;
    result(x<0) = leak;
end