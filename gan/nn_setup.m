function net = nn_setup(net)
    net.t = 0;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 10^(-8);
    for l = 2 : numel(net.layers)
        input_shape = net.layers{l-1}.output_shape;
        switch net.layers{l}.type
            case 'conv2d'
                net.layers{l} = setup_conv2d_layer(input_shape, net.layers{l});
            case 'sub_sampling'
                net.layers{l} = setup_sub_sampling_layer(input_shape, net.layers{l});
            case 'fully_connect'
                net.layers{l} = setup_fully_connect_layer(input_shape, net.layers{l});
            case 'reshape'
                net.layers{l} = setup_reshape_layer(input_shape, net.layers{l});
            case 'conv2d_transpose'
                net.layers{l} = setup_conv2d_transpose_layer(input_shape, net.layers{l});
            case 'atrous_conv2d'
                net.layers{l} = setup_atrous_conv2d_layer(input_shape, net.layers{l});
            otherwise
                error('wrong layer type:%s', net.layers{l}.type)
        end
    end
end