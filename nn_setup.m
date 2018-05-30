function net = nn_setup(net)
    net.t = 0;
    net.beta1 = 0.9;
    net.beta2 = 0.999;
    net.epsilon = 10^(-8);
    for l = 2 : numel(net.layers)  
        if strcmp(net.layers{l}.type, 'conv2d')
            % stride should be 1,and filter size should be odd
            net.layers{l}.input_shape =  net.layers{l-1}.output_shape;
            input_shape = net.layers{l}.input_shape;
            if numel(input_shape)==3
                input_maps = 1;
            elseif numel(input_shape)==4
                input_maps = input_shape(end);
            else
                error('sss')
            end
            net.layers{l} = check_conv2d_layer(net.layers{l});
            net.layers{l}.filter = normrnd(0, 0.02, net.layers{l}.kernel_size, net.layers{l}.kernel_size, input_maps, net.layers{l}.output_maps);
            net.layers{l}.biases = normrnd(0, 0.02, 1, net.layers{l}.output_maps);
            net.layers{l}.filter_m = 0;
            net.layers{l}.filter_v = 0;
            net.layers{l}.biases_m = 0;
            net.layers{l}.biases_v = 0;
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
            net.layers{l}.input_shape =  net.layers{l-1}.output_shape;
            input_shape = net.layers{l}.input_shape;
            if numel(input_shape)==3
                input_maps = 1;
            elseif numel(input_shape)==4
                input_maps = input_shape(end);
            else
                error('sss')
            end
            kernel_size = net.layers{l}.kernel_size;
            output_shape = net.layers{l}.output_shape;
            net.layers{l} = check_conv2d_transpose_layer(net.layers{l});
            net.layers{l}.filter = normrnd(0, 0.02, kernel_size, kernel_size, input_maps, output_shape(end));
            net.layers{l}.biases = normrnd(0, 0.02, 1, output_shape(end));
            net.layers{l}.filter_m = 0;
            net.layers{l}.filter_v = 0;
            net.layers{l}.biases_m = 0;
            net.layers{l}.biases_v = 0;
        % ---------- atrous conv2d
        elseif strcmp(net.layers{l}.type, 'atrous_conv2d')
            net.layers{l}.input_shape =  net.layers{l-1}.output_shape;
            input_shape = net.layers{l}.input_shape;
            if numel(input_shape)==3
                input_maps = 1;
            elseif numel(input_shape)==4
                input_maps = input_shape(end);
            else
                error('sss')
            end
            net.layers{l}.filter = normrnd(0, 0.02, net.layers{l}.kernel_size, net.layers{l}.kernel_size, input_maps, net.layers{l}.output_maps);
            net.layers{l}.biases = normrnd(0, 0.02, 1, net.layers{l}.output_maps);
            net.layers{l}.filter_m = 0;
            net.layers{l}.filter_v = 0;
            net.layers{l}.biases_m = 0;
            net.layers{l}.biases_v = 0;
        end
    end
end