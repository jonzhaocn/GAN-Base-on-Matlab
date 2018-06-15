% reference: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
function net = nn_applygrads_adam(net, learning_rate)
    net.t = net.t+1;
    beta1 = net.beta1;
    beta2 = net.beta2;
    lr = learning_rate * sqrt(1-net.beta2^net.t) / (1-net.beta1^net.t);
    for l = 2 : numel(net.layers)
        switch net.layers{l}.type
            case {'conv2d','conv2d_transpose','atrous_conv2d'}
                dfilter = net.layers{l}.dfilter;
                dbiases = net.layers{l}.dbiases;
                net.layers{l}.filter_m = beta1 * net.layers{l}.filter_m + (1-beta1) * dfilter;
                net.layers{l}.filter_v = beta2 * net.layers{l}.filter_v + (1-beta2) * (dfilter .* dfilter);
                net.layers{l}.filter = net.layers{l}.filter - lr * net.layers{l}.filter_m ./ (sqrt(net.layers{l}.filter_v) + net.epsilon);
                net.layers{l}.biases_m = beta1 * net.layers{l}.biases_m + (1-beta1) * dbiases;
                net.layers{l}.biases_v = beta2 * net.layers{l}.biases_v + (1-beta2) * (dbiases .* dbiases);
                net.layers{l}.biases = net.layers{l}.biases - lr * net.layers{l}.biases_m ./ (sqrt(net.layers{l}.biases_v) + net.epsilon);
            case 'fully_connect'
                dweights = net.layers{l}.dweights;
                dbiases = net.layers{l}.dbiases;
                net.layers{l}.weights_m = beta1 * net.layers{l}.weights_m + (1-beta1) * dweights;
                net.layers{l}.weights_v = beta2 * net.layers{l}.weights_v + (1-beta2) * (dweights .* dweights);
                net.layers{l}.weights = net.layers{l}.weights - lr* net.layers{l}.weights_m ./ (sqrt(net.layers{l}.weights_v) + net.epsilon);
                net.layers{l}.biases_m = beta1 * net.layers{l}.biases_m + (1-beta1) * dbiases;
                net.layers{l}.biases_v = beta2 * net.layers{l}.biases_v + (1-beta2) * (dbiases .* dbiases);
                net.layers{l}.biases = net.layers{l}.biases - lr * net.layers{l}.biases_m ./ (sqrt(net.layers{l}.biases_v) + net.epsilon);
            case {'reshape', 'sub_sampling'}
                continue
            otherwise
                error('wrong layer type %s', net.layers{l}.type)
        end
    end
end