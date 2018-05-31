function layer = setup_conv2d_layer(input_shape, layer)
    % ---------check------------
    required_fields = {'type', 'kernel_size', 'output_maps', 'padding'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    % -------- init --------------
    batch_size = input_shape(1);
    in_height = input_shape(2);
    in_width = input_shape(3);
    kernel_size = layer.kernel_size;
    output_maps = layer.output_maps;
    layer.input_shape = input_shape;
    
    % --------- setting---------
    if strcmp(layer.padding, 'valid')
        out_height = in_height - kernel_size + 1;
        out_width = in_width - kernel_size + 1;
    elseif strcmp(layer.padding, 'same')
        out_height = in_height;
        out_width = in_width;
    else
        error('padding of conv2d layer should be valid or same')
    end
    layer.output_shape = [batch_size, out_height, out_width, layer.output_maps];
    
    if numel(input_shape)==3
        input_maps = 1;
    elseif numel(input_shape)==4
        input_maps = input_shape(end);
    else
        error('wrong input shape in conv2d layer, dims of input should be 3 or 4,now input shape is [%s]', num2str(input_shape))
    end
    layer.filter = normrnd(0, 0.02, kernel_size, kernel_size, input_maps, output_maps);
    layer.biases = normrnd(0, 0.02, 1, output_maps);
    layer.filter_m = 0;
    layer.filter_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end