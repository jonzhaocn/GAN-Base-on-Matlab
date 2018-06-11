function net = nn_applygrads_sgd(net, learning_rate)
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'conv2d') || strcmp(net.layers{l}.type, 'conv2d_transpose') || strcmp(net.layers{l}.type, 'atrous_conv2d')
            net.layers{l}.filter = net.layers{l}.filter - learning_rate * net.layers{l}.dfilter;
            net.layers{l}.biases = net.layers{l}.biases - learning_rate * net.layers{l}.dbiases;
        elseif strcmp(net.layers{l}.type, 'fully_connect')
            net.layers{l}.weights = net.layers{l}.weights - learning_rate * net.layers{l}.dweights;
            net.layers{l}.biases = net.layers{l}.biases - learning_rate * net.layers{l}.dbiases;
        elseif strcmp(net.layers{l}.type, 'reshape')
            continue;
        elseif strcmp(net.layers{l}.type, 'sub_sampling')
            continue;
        else 
            error('wrong layer type')
        end
    end
end