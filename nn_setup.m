function net = nn_setup(net)
    net.t = 0;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 10^(-8);
    for l = 2 : numel(net.layers)  
        if strcmp(net.layers{l}.type, 'conv2d')
            % stride should be 1,and filter size should be odd
            net.layers{l} = setup_conv2d_layer(net.layers{l-1}.output_shape, net.layers{l});
            
        elseif strcmp(net.layers{l}.type, 'sub_sampling')
            net.layers{l}.input_shape = net.layers{l-1}.output_shape;
            net.layers{l}.output_shape = net.layers{l}.input_shape;
            net.layers{l}.output_shape(2:3) = floor(net.layers{l}.output_shape(2:3)/net.layers{l}.scale);
        % ------------ fully connect
        elseif strcmp(net.layers{l}.type, 'fully_connect')
            net.layers{l}.input_shape =  net.layers{l-1}.output_shape;
            input_shape = net.layers{l}.input_shape;
            if numel(input_shape)~=2
                error('input shape of fully connect is wrong');
            end
            net.layers{l}.weights = normrnd(0, 0.02, input_shape(end), net.layers{l}.output_shape(end));
            net.layers{l}.biases = normrnd(0, 0.02, 1, net.layers{l}.output_shape(end));
            net.layers{l}.weights_m = 0;
            net.layers{l}.weights_v = 0;
            net.layers{l}.biases_m = 0;
            net.layers{l}.biases_v = 0;
        % ------------- reshape
        elseif strcmp(net.layers{l}.type, 'reshape')
            net.layers{l}.input_shape =  net.layers{l-1}.output_shape;
            input_shape = net.layers{l}.input_shape;
            if prod(input_shape) ~= prod(net.layers{l}.output_shape)
                error(['reshape outpushape error, input shape is: ', num2str(input_shape)])
            end
        % -------------- conv transpose
        elseif strcmp(net.layers{l}.type, 'conv2d_transpose')
           net.layers{l} = setup_conv2d_transpose_layer(net.layers{l-1}.output_shape, net.layers{l});
        % ---------- atrous conv2d
        elseif strcmp(net.layers{l}.type, 'atrous_conv2d')
            net.layers{l} = setup_atrous_conv2d_layer(net.layers{l-1}.output_shape, net.layers{l});
        else
            error('error layer type')
        end
    end
end