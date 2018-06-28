function net = nn_applygrads_sgd(net, learning_rate)
    for l = 2 : numel(net.layers)
        switch net.layers{l}.type
            case {'conv2d', 'conv2d_transpose', 'atrous_conv2d', 'fully_connect', 'batch_norm'}
                net.layers{l}.weights = net.layers{l}.weights - learning_rate * net.layers{l}.dweights;
                net.layers{l}.biases = net.layers{l}.biases - learning_rate * net.layers{l}.dbiases;
            case {'reshape', 'sub_sampling'}
                continue
            otherwise
                error('wrong layer type %s', net.layers{l}.type)
        end
    end
end