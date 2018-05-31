function layer = setup_sub_sampling_layer(input_shape, layer)
    % ------ check
    required_fields = {'type', 'scale'};
    optional_fields = {};
    check_layer_field_names(layer, required_fields, optional_fields);
    % -------- set
    layer.input_shape = input_shape;
    layer.output_shape = input_shape;
    layer.output_shape(2:3) = floor(layer.output_shape(2:3)/layer.scale);
end