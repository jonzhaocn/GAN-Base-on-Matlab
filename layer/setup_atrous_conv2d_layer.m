function layer = setup_atrous_conv2d_layer(input_shape, layer)
    %---------check
    required_fields = {'type', 'kernel_size', 'rate', 'output_maps', 'padding'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    %--------init
    kernel_size = layer.kernel_size;
    rate = layer.rate;
    batch_size = input_shape(4);
    output_maps = layer.output_maps;
    %--------setting
    layer.input_shape = input_shape;
    input_maps = input_shape(3);
    % --
    ks = kernel_size+(kernel_size-1)*(rate-1);
    switch layer.padding
        case 'valid'
            out_height = input_shape(1) - ks + 1;
            out_width = input_shape(2) - ks + 1;
            layer.padding_shape = [ks-1, ks-1, ks-1, ks-1];
        case 'same'
            out_height = input_shape(1);
            out_width = input_shape(2);
            if mod(kernel_size,2)==0
                error('atrous conv2d layer, padding is same, kernel size should be a odd');
            end
            layer.padding_shape = [floor(ks/2), floor(ks/2), floor(ks/2), floor(ks/2)];
        otherwise
            error('padding of atours conv2d layer should be valid or same')
    end
    layer.output_shape = [out_height, out_width, output_maps, batch_size];
    layer.weights = normrnd(0, 0.01, kernel_size, kernel_size, input_maps, output_maps);
    layer.biases = normrnd(0, 0.01, output_maps, 1);
    layer.weights_m = 0;
    layer.weights_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end