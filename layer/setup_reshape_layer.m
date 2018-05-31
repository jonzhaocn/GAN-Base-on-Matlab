function layer = setup_reshape_layer(input_shape, layer)
    % ------check-------
    required_fields = {'type', 'output_shape'};
    optional_fields = {};
    check_layer_field_names(layer, required_fields, optional_fields);
    % ------setting----------
    layer.input_shape =  input_shape;
    if prod(input_shape) ~= prod(layer.output_shape)
        error('cannot reshape [%s] to [%s]', num2str(input_shape), num2str(layer.output_shape))
    end
end