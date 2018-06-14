function result = get_error_term_from_reshape_layer(back_layer)
    result = reshape_operation(back_layer.d, back_layer.input_shape);
end