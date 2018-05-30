function layer = check_conv2d_layer(layer)
    % -------- init --------------
    batch_size = layer.input_shape(1);
    in_height = layer.input_shape(2);
    in_width = layer.input_shape(3);
    kernel_size = layer.kernel_size;
    % --------- calculate---------
    if strcmp(layer.padding, 'valid')
        out_height = in_height - kernel_size + 1;
        out_width = in_width - kernel_size + 1;
    elseif strcmp(layer.padding, 'same')
        out_height = in_height;
        out_width = in_width;
    else
        error('padding should be valid or same')
    end
    layer.output_shape = [batch_size, out_height, out_width, layer.output_maps];
end