function result = get_error_term_from_conv2d_layer(back_layer)
    weights = back_layer.weights;
    d = back_layer.d;
    batch_size = size(d, 4);
    result = zeros([back_layer.input_shape(1:end-1), batch_size]);
    % because we use convn, we should padding height and width in d,
    % and convn padding which is valid will become conv2d
    d = padding_height_width_in_array(d, back_layer.padding_shape);
    % ---
    for ii = 1:size(weights,3)
        result(:,:,ii,:) =  convn(d, flipall( squeeze( weights(:,:,ii,:) )), "valid");
    end
    
end