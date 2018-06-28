function net = nn_ff(net, x)
    n = numel(net.layers);
    net.layers{1}.a = x;
    for l = 2 : n   %  for each layer
		input = net.layers{l-1}.a;
        switch net.layers{l}.type
            case 'conv2d'
                net.layers{l}.z = conv2d(input, net.layers{l});
                for i=1:net.layers{l}.output_maps
                    net.layers{l}.z(:,:,i,:) = net.layers{l}.z(:,:,i,:) + net.layers{l}.biases(i, 1);
                end
            case 'sub_sampling'
                net.layers{l}.z = sub_sample(input, net.layers{l}.scale);
            case 'fully_connect'
               net.layers{l}.z = net.layers{l}.weights' * input  + repmat(net.layers{l}.biases, 1, size(input,2));
            case 'reshape'
                net.layers{l}.z = reshape_operation(input, net.layers{l}.output_shape);
            case 'conv2d_transpose'
                net.layers{l}.z = conv2d_transpose(input, net.layers{l});
                for i=1:net.layers{l}.output_shape(3)
                    net.layers{l}.z(:,:,i,:) = net.layers{l}.z(:,:,i,:) + net.layers{l}.biases(i, 1);
                end
            case 'atrous_conv2d'
                net.layers{l}.z = atrous_conv2d(input, net.layers{l});
                for i=1:net.layers{l}.output_maps
                    net.layers{l}.z(:,:,i,:) = net.layers{l}.z(:,:,i,:) + net.layers{l}.biases(i, 1);
                end
            case 'batch_norm'
               net.layers{l} = batch_norm(input, net.layers{l});
            otherwise
                error('wrong layer type:%s', net.layers{l}.type)
        end
        net.layers{l}.a = activate_z(net.layers{l});
    end
end