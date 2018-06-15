function layer = setup_conv2d_layer(input_shape, layer)
    % ---------check------------
    required_fields = {'type', 'kernel_size', 'output_maps', 'padding'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    % -------- init --------------
    in_height = input_shape(1);
    in_width = input_shape(2);
    input_maps = input_shape(3);
    batch_size = input_shape(4);
    kernel_size = layer.kernel_size;
    output_maps = layer.output_maps;
    layer.input_shape = input_shape;
    
    % --------- setting---------
    switch layer.padding
        case 'valid'
            out_height = in_height - kernel_size + 1;
            out_width = in_width - kernel_size + 1;
            layer.padding_shape = [kernel_size-1, kernel_size-1, kernel_size-1, kernel_size-1];
        case 'same'
            out_height = in_height;
            out_width = in_width;
            if mod(kernel_size,2)==0
                error('conv2d padding is same, kernel size should be a odd')
            end
            layer.padding_shape = [floor(kernel_size/2), floor(kernel_size/2), floor(kernel_size/2), floor(kernel_size/2)];
        otherwise
            error('padding of conv2d layer should be valid or same')
    end
    layer.output_shape = [out_height, out_width, layer.output_maps, batch_size];
    layer.filter = normrnd(0, 0.01, kernel_size, kernel_size, input_maps, output_maps);
    layer.biases = normrnd(0, 0.01, 1, output_maps);
    layer.filter_m = 0;
    layer.filter_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end