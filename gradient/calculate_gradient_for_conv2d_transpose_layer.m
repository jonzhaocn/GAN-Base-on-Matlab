function [dfilter, dbiases] = calculate_gradient_for_conv2d_transpose_layer(front_a, layer)
    d = layer.d;
    filter = layer.filter;
    dfilter = zeros(size(filter));
    dbiases = zeros(size(layer.biases));
    stride = layer.stride;
    if strcmp(layer.padding, 'valid')
        if stride == 1
            % empty
        else
            front_a = insert_zeros_into_array(front_a, layer.stride);
        end
        front_a = permute(front_a,[2,3,1,4]);
        d = permute(d,[2,3,1,4]);
        for jj = 1:size(filter,4)
            d_j = d(:,:,:,jj);
            for ii = 1:size(filter,3)
                % 这里需要剪裁
                dfilter(:,:,ii,jj) = convn(d_j, flipall(front_a(:,:,:,ii)), "valid") / size(d, 3);
            end
            dbiases(1, jj) = sum(d_j(:)) / size(d, 3);
        end
    elseif strcmp(layer.padding, 'same')
        if stride == 1
            % padding d
            p_top = layer.padding_shape(1);
            p_left = layer.padding_shape(2);
            d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
        else
            % insert 0 into a
            front_a = insert_zeros_into_array(front_a, layer.stride);
            p_top = layer.padding_shape(1);
            p_left = layer.padding_shape(2);
            p_bottom = p_top - layer.a_padding_shape(1);
            p_right = p_left - layer.a_padding_shape(2);
            d = padding_height_width_in_array(d, p_top, p_bottom, p_left, p_right);
        end
        front_a = permute(front_a,[2,3,1,4]);
        d = permute(d,[2,3,1,4]);
        % conv
        for jj = 1:size(filter,4)
            d_j = d(:,:,:,jj);
            for ii = 1:size(filter,3)
                dfilter(:,:,ii,jj) = squeeze(convn(d_j, flipall(front_a(:,:,:,ii)), "valid")) / size(d, 3);
            end
            dbiases(1, jj) = sum(d_j(:)) / size(d, 3);
        end
    end
end