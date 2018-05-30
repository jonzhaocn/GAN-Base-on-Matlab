function layer = setup_atrous_conv2d_layer(input_shape, layer)
    layer.input_shape = input_shape;
    kernel_size = layer.kernel_size;
    rate = layer.rate;
    batch_size = input_shape(1);
    output_maps = layer.output_maps;
    if numel(input_shape)==3
        input_maps = 1;
    elseif numel(input_shape)==4
        input_maps = input_shape(end);
    else
        error('wrong input shape in atrous conv2d ayer')
    end
    if strcmp(layer.padding, "valid")
        out_height = input_shape(2) - (kernel_size+(kernel_size-1)*(rate-1)) + 1;
        out_width = input_shape(3) - (kernel_size + (kernel_size-1)*(rate-1)) + 1; 
    elseif strcmp(layer.padding, "same")
        out_height = input_shape(2);
        out_width = input_shape(3);
    else
        error('padding should be valid or same')
    end
    layer.output_shape = [batch_size, out_height, out_width, output_maps];
    layer.filter = normrnd(0, 0.02, kernel_size, kernel_size, input_maps, output_maps);
    layer.biases = normrnd(0, 0.02, 1, layer.output_maps);
    layer.filter_m = 0;
    layer.filter_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end