function output = atrous_conv2d(input, layer)
    weights = layer.weights;
    [in_height, in_width, in_channel, batch_size] = size(input);
    [weights_height, weights_width, weights_in_channel, out_channel] = size(weights);
    weights = insert_zeros_into_array(weights, layer.rate);
    switch layer.padding
        case 'valid'
            out_height = in_height - weights_height + 1;
            out_width = in_width - weights_width + 1;
        case 'same'
            out_height = in_height;
            out_width = in_width;
            input = padding_height_width_in_array(input, layer.padding_shape);
        otherwise
            error('padding of atours conv2d should be valid or same');
    end
    output = zeros(out_height, out_width, out_channel, batch_size);
    for jj = 1:out_channel
        output(:,:,jj,:) = convn(input, flip(weights(:,:,:,jj), 3), "valid");
    end
end