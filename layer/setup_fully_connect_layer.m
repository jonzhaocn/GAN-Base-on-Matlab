function layer = setup_fully_connect_layer(input_shape, layer)
    % ----- check
    required_fields = {'type', 'output_shape'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    % ----- setting
    layer.input_shape =  input_shape;
    if numel(input_shape)~=2
        error('input shape of fully connect is wrong, dims of input_shape should be 2, input shape is [%s]', num2str(input_shape));
    end
    layer.weights = normrnd(0, 0.01, input_shape(1),layer.output_shape(1));
    layer.biases = normrnd(0, 0.01, layer.output_shape(1), 1);
    layer.weights_m = 0;
    layer.weights_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end