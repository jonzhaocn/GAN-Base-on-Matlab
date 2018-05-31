function result = get_error_term_from_atrous_conv2d_layer(back_layer)
    filter = insert_zeros_into_filter(back_layer.filter, back_layer.rate);
    d = back_layer.d;
    result = zeros([size(d, 1), back_layer.input_shape(2:end)]);
    p_top = back_layer.padding_shape(1);
    p_left = back_layer.padding_shape(2);
    d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
    % after been permuted,result is [height,width,batch_size,channel],
    % d is [height,width,channel,batch_size]
    result = permute(result, [2,3,1,4]);
    d = permute(d, [2,3,4,1]);
    for ii = 1:size(filter,3)
        result(:,:,:,ii) =  squeeze(convn(d(:,:,:,:), flip( squeeze( filter(:,:,ii,:) ),3 ), "valid"));
    end
    result = permute(result, [3,1,2,4]);
end