function result = get_error_term_from_conv2d_transpose_layer(back_layer)
    weights = back_layer.weights;
    % after been permuted, d is [height,width,batch_size,channel]
    d = back_layer.d;
    [d_height, d_width, ~, batch_size] = size(d);
    [weights_height, weights_width, in_channel, ~] = size(weights);
    switch back_layer.padding
        case 'valid'
            % ---------------- valid stride==1
            if back_layer.stride == 1
                result = zeros([back_layer.input_shape(1:end-1), batch_size]);
                for ii = 1:size(weights,3)
                    result(:,:,ii,:) = convn(d, flipall(squeeze(weights(:,:,ii,:))), "valid");
                end
                % ----------------- valid stride>1
            else
                % temp [height, width, channel, batch_size]
                temp = zeros(d_height-weights_height+1, d_width-weights_width+1, in_channel, batch_size);
                for ii = 1:size(weights,3)
                    temp(:,:,ii,:) = convn(d, flipall(squeeze(weights(:,:,ii,:))), "valid");
                end
                result = temp(1:back_layer.stride:end,1:back_layer.stride:end,:,:);
            end
        case 'same'
            if back_layer.stride == 1
                d = padding_height_width_in_array(d, back_layer.padding_shape);
            else
                d = padding_height_width_in_array(d, back_layer.padding_shape - back_layer.a_padding_shape);
            end
            % --
            [d_height, d_width, ~, ~] = size(d);
            temp = zeros(d_height-weights_height+1, d_width-weights_width+1, in_channel, batch_size);
            for ii = 1:size(weights,3)
                temp(:,:,ii,:) = convn(d, flipall(squeeze(weights(:,:,ii,:))), "valid");
            end
            result = temp(1:back_layer.stride:end, 1:back_layer.stride:end, :, :);
        otherwise
            error('padding only support valid or same')
    end
end