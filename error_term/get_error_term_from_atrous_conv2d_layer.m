% get error term for atrous conv2d layer with respect to current layer's output
function result = get_error_term_from_atrous_conv2d_layer(back_layer)
    weights = insert_zeros_into_filter(back_layer.weights, back_layer.rate);
    d = back_layer.d;
    batch_size = size(d, 4);
    result = zeros([back_layer.input_shape(1:end-1), batch_size]);
    d = padding_height_width_in_array(d, back_layer.padding_shape);
    % --
    for ii = 1:size(weights,3)
        result(:, :, ii, :) =  convn(d, flipall( squeeze( weights(:,:,ii,:) )), "valid");
    end
end