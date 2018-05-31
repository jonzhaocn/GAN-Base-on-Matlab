function result = get_error_term_from_reshape_layer(back_layer)
    result = reshape(back_layer.d, [size(back_layer.d,1), back_layer.input_shape(2:end)]);
end