function result = get_error_term_from_atrous_conv2d_layer(back_layer)
    filter = insert_zeros_into_filter(back_layer.filter, back_layer.rate);
    d = back_layer.d;
    batch_size = size(d, 4);
    result = zeros([back_layer.input_shape(1:end-1), batch_size]);
    p_top = back_layer.padding_shape(1);
    p_left = back_layer.padding_shape(2);
    d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
    % --
    for ii = 1:size(filter,3)
        result(:, :, ii, :) =  convn(d, flipall( squeeze( filter(:,:,ii,:) )), "valid");
    end
end