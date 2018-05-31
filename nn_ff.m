function net = nn_ff(net, x)
    n = numel(net.layers);
    net.layers{1}.a = x;
    
    for l = 2 : n   %  for each layer
        % --------conv2d
        if strcmp(net.layers{l}.type, 'conv2d')
            input = net.layers{l-1}.a;
            % conv2d(input, filter, padding)
            net.layers{l}.z = conv2d(input, net.layers{l});
            % add biases
            for i=1:net.layers{l}.output_maps
                net.layers{l}.z(:,:,:,i) = net.layers{l}.z(:,:,:,i) + net.layers{l}.biases(1,i);
            end
        % -----------sub sampling
        elseif strcmp(net.layers{l}.type, 'sub_sampling')
            input = net.layers{l-1}.a;
            net.layers{l}.z = sub_sample(input, net.layers{l}.scale);
        % -----------fully connect
        elseif strcmp(net.layers{l}.type, 'fully_connect')
            input = net.layers{l-1}.a;
            net.layers{l}.z = input * net.layers{l}.weights + repmat(net.layers{l}.biases, size(input, 1), 1);
        % ----------- reshape 
        elseif strcmp(net.layers{l}.type, 'reshape')
            input = net.layers{l-1}.a;
            net.layers{l}.z = reshape(input, [size(input,1), net.layers{l}.output_shape(2:end)]);
        % ----------- conv2d transpose
        elseif strcmp(net.layers{l}.type, 'conv2d_transpose')
            input = net.layers{l-1}.a;
            % conv2d_transpose(input, filter, output_shape, padding)
            % z [batch_size,height,width,out_channel]
            net.layers{l}.z = conv2d_transpose(input, net.layers{l});
            % add biases
            for i=1:net.layers{l}.output_shape(end)
                net.layers{l}.z(:,:,:,i) = net.layers{l}.z(:,:,:,i) + net.layers{l}.biases(1,i);
            end
        % --------- atrous conv2d
        elseif strcmp(net.layers{l}.type, 'atrous_conv2d')
            input = net.layers{l-1}.a;
            % atrous_conv2d(input, filter, rate, padding)
            net.layers{l}.z = atrous_conv2d(input, net.layers{l});
            % add biases
            for i=1:net.layers{l}.output_maps
                net.layers{l}.z(:,:,:,i) = net.layers{l}.z(:,:,:,i) + net.layers{l}.biases(1,i);
            end
        end
        net.layers{l}.a = activate_z(net.layers{l});
    end
end