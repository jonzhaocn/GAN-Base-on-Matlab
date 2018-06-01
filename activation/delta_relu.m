% The derivative of relu with respect to x
function output = delta_relu(x)
    output = x;
    output(x>=0) = 1;
    output = max(output,0);
end