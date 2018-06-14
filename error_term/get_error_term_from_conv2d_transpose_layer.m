function result = get_error_term_from_conv2d_transpose_layer(back_layer)
    filter = back_layer.filter;
    % after been permuted, d is [height,width,batch_size,channel]
    d = back_layer.d;
    [d_height, d_width, ~, batch_size] = size(d);
    [filter_height, filter_width, in_channel, ~] = size(filter);
    switch back_layer.padding
        case 'valid'
            % ---------------- valid stride==1
            if back_layer.stride == 1
                result = zeros([back_layer.input_shape(1:end-1), batch_size]);
                for ii = 1:size(filter,3)
                    result(:,:,ii,:) = convn(d, flipall(squeeze(filter(:,:,ii,:))), "valid");
                end
                % ----------------- valid stride>1
            else
                % temp [height, width, channel, batch_size]
                temp = zeros(d_height-filter_height+1, d_width-filter_width+1, in_channel, batch_size);
                for ii = 1:size(filter,3)
                    temp(:,:,ii,:) = convn(d, flipall(squeeze(filter(:,:,ii,:))), "valid");
                end
                result = temp(1:back_layer.stride:end,1:back_layer.stride:end,:,:);
            end
        case 'same'
            if back_layer.stride == 1
                p_top = back_layer.padding_shape(1);
                p_left = back_layer.padding_shape(2);
                d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
            else
                p_top = back_layer.padding_shape(1);
                p_left = back_layer.padding_shape(2);
                p_bottom = p_top - back_layer.a_padding_shape(1);
                p_right = p_left - back_layer.a_padding_shape(2);
                d = padding_height_width_in_array(d, p_top, p_bottom, p_left, p_right);
            end
            % --
            [d_height, d_width, ~, ~] = size(d);
            temp = zeros(d_height-filter_height+1, d_width-filter_width+1, in_channel, batch_size);
            for ii = 1:size(filter,3)
                temp(:,:,ii,:) = convn(d, flipall(squeeze(filter(:,:,ii,:))), "valid");
            end
            result = temp(1:back_layer.stride:end, 1:back_layer.stride:end, :, :);
        otherwise
            error('padding only support valid or same')
    end
end