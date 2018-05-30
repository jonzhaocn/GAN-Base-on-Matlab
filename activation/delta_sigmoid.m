% % The derivative of sigmoid with respect to x
%http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C
function output = delta_sigmoid(x)
    output = sigmoid(x).*(1-sigmoid(x));
end