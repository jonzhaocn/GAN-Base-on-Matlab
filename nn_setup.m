function net = nn_setup(net)
    net.t = 0;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 10^(-8);
    for l = 2 : numel(net.layers)
        input_shape = net.layers{l-1}.output_shape;
        if strcmp(net.layers{l}.type, 'conv2d')
            % stride should be 1,and filter size should be odd
            net.layers{l} = setup_conv2d_layer(input_shape, net.layers{l});
            
        elseif strcmp(net.layers{l}.type, 'sub_sampling')
            net.layers{l} = setup_sub_sampling_layer(input_shape, net.layers{l});
            
        % ------------ fully connect
        elseif strcmp(net.layers{l}.type, 'fully_connect')
            net.layers{l} = setup_fully_connect_layer(input_shape, net.layers{l});
        % ------------- reshape
        elseif strcmp(net.layers{l}.type, 'reshape')
            net.layers{l} = setup_reshape_layer(input_shape, net.layers{l});
        % -------------- conv transpose
        elseif strcmp(net.layers{l}.type, 'conv2d_transpose')
           net.layers{l} = setup_conv2d_transpose_layer(input_shape, net.layers{l});
        % ---------- atrous conv2d
        elseif strcmp(net.layers{l}.type, 'atrous_conv2d')
            net.layers{l} = setup_atrous_conv2d_layer(input_shape, net.layers{l});
        else
            error(['wrong layer type:', net.layers{l}.type])
        end
    end
end