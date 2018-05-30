% relu
% if x>=0 x
% if x<0 0
function output = relu(x)
    output = max(x, 0);
end