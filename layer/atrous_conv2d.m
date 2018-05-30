function output = atrous_conv2d(input, layer)
    filter = layer.filter;
    [batch_size, in_height, in_width, in_channel] = size(input);
    [filter_height, filter_width, filter_in_channel, out_channel] = size(filter);
    filter = insert_zeros_into_filter(filter, layer.rate);
    if strcmp(layer.padding, 'valid')
        out_height = in_height - size(filter,1) + 1;
        out_width = in_width - size(filter,2) + 1;
    elseif strcmp(layer.padding, 'same')
        out_height = in_height;
        out_width = in_width;
        p_top = floor(size(filter,1)/2);
        p_left = floor(size(filter,2)/2);
        input = padding_height_width_in_array(input, p_top, p_top, p_left, p_left);
    else
        error('padding should be valid or same');
    end
    output = zeros(out_height, out_width, batch_size, out_channel);
    % after permuting, input become [height, width, in_channel, batch_size]
    input = permute(input,[2,3,4,1]);
    for jj = size(filter,4)
        output(:,:,:,jj) = squeeze(convn(input, flip(filter(:,:,:,jj), 3), "valid"));
    end
    output = permute(output, [3,1,2,4]);
end