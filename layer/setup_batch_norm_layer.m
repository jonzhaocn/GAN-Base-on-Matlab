function layer = setup_batch_norm_layer(input_shape, layer)
    %---------check
    required_fields = {'type'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    %--------init
    if numel(input_shape)==2
        input_maps = 1;
    elseif numel(input_shape)==4
        input_maps = input_shape(3);
    else 
        error('wrong input shape')
    end
    %--------setting
    layer.input_shape = input_shape;
    layer.output_shape = input_shape;
    layer.epslion = 1e-4;
    % --
    layer.weights = normrnd(0, 0.01, input_maps, 1);
    layer.biases = normrnd(0, 0.01, input_maps, 1);
    layer.weights_m = 0;
    layer.weights_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end