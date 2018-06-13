function result = get_error_term_from_conv2d_transpose_layer(back_layer)
    filter = back_layer.filter;
    % after been permuted, d is [height,width,batch_size,channel]
    d = back_layer.d;
    if strcmp(back_layer.padding, "valid")
        % ---------------- valid stride==1
        if back_layer.stride == 1
            result = zeros([size(d,1), back_layer.input_shape(2:end)]);
            result = permute(result, [2,3,1,4]);
            for ii = 1:size(filter,3)
                result(:,:,:,ii) = squeeze(convn(d, flipall(squeeze(filter(:,:,ii,:))), "valid"));
            end
        % ----------------- valid stride>1
        else
            % temp [batch_size, height, width, channel]
            temp = zeros(size(d,1), size(d,2)-size(filter,1)+1, size(d,3)-size(filter,2)+1, size(filter,3));
            % after been permute,temp become [height, width, batch_size, channel]
            temp = permute(temp, [2,3,1,4]);
            % after been permute,d become [height, width, channel, batch_size]
            d = permute(d, [2,3,4,1]);
            for ii = 1:size(filter,3)
                temp(:,:,:,ii) = squeeze(convn(d, flipall(squeeze(filter(:,:,ii,:))), "valid"));
            end
            result = temp(1:back_layer.stride:end,1:back_layer.stride:end,:,:);
        end
    % ---------------same------------------
    elseif strcmp(back_layer.padding, "same")
        % padding d
        if back_layer.stride == 1
            p_top = back_layer.padding_shape(1);
            p_left = back_layer.padding_shape(2);
            d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
        elseif back_layer.stride == 2
            p_top = back_layer.padding_shape(1);
            p_left = back_layer.padding_shape(2);
            p_bottom = p_top - back_layer.a_padding_shape(1);
            p_right = p_left - back_layer.a_padding_shape(2);
            d = padding_height_width_in_array(d, p_top, p_bottom, p_left, p_right);
        end
        temp = zeros(size(d,1), size(d,2)-size(filter,1)+1, size(d,3)-size(filter,2)+1, size(filter,3));
        temp = permute(temp, [2,3,1,4]);
        % permute
        d = permute(d, [2,3,4,1]);
        % conv 
        for ii = 1:size(filter,3)
            temp(:,:,:,ii) = squeeze(convn(d, flipall(squeeze(filter(:,:,ii,:))), "valid"));
        end
        result = temp(1:back_layer.stride:end, 1:back_layer.stride:end, :, :);
    end
    result = permute(result,[3,1,2,4]);
end