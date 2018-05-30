function result = get_error_term_from_sub_sampling_layer(back_layer)
    d = back_layer.d;
    if ndims(d)== 3
        expand_shape = [1 back_layer.scale back_layer.scale];
    elseif ndims(d)== 4
        expand_shape = [1 back_layer.scale back_layer.scale 1];
    else
        error('xxx')
    end
    result = expand(back_layer.d, expand_shape) / back_layer.scale ^ 2;
end