function result = get_error_term_from_fully_connect_layer(back_layer)
    result = back_layer.d * back_layer.weights';
end