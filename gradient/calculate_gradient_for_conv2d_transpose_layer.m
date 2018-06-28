function [dweights, dbiases] = calculate_gradient_for_conv2d_transpose_layer(front_a, layer)
    % ---
    d = layer.d;
    weights = layer.weights;
    dweights = zeros(size(weights));
    dbiases = zeros(size(layer.biases));
    stride = layer.stride;
    batch_size = size(d, 4);
    % ----
    switch layer.padding
        case 'valid'
            if stride == 1
                % empty
            else
                front_a = insert_zeros_into_array(front_a, layer.stride);
            end
        case 'same'
            if stride == 1
                % padding d
                d = padding_height_width_in_array(d, layer.padding_shape);
            else
                % insert 0 into a
                front_a = insert_zeros_into_array(front_a, layer.stride);
                d = padding_height_width_in_array(d, layer.padding_shape-layer.a_padding_shape);
            end
        otherwise
            error('padding only support valid or same')
    end
    for jj = 1:size(weights,4)
        d_j = squeeze(d(:,:,jj,:));
        for ii = 1:size(weights,3)
            dweights(:,:,ii,jj) = squeeze(convn(d_j, flipall( squeeze(front_a(:,:,ii,:)) ), "valid")) / batch_size;
        end
        dbiases(jj, 1) = sum(d_j(:)) / batch_size;
    end
end