function [dweights, dbiases] = calculate_gradient_for_atrous_conv2d_layer(front_a, layer)
    d = layer.d;
    weights = layer.weights;
    weights = insert_zeros_into_filter(weights, layer.rate);
    dweights = zeros(size(weights));
    dbiases = zeros(size(layer.biases));
    batch_size = size(d, 4);
    % padding
    d = padding_height_width_in_array(d, layer.padding_shape);
    
    for jj = 1:size(weights,4) %output channel
        d_j = squeeze(d(:,:,jj,:));
        for ii=1:size(weights,3) % input channel
            dweights(:,:,ii,jj) =  squeeze(convn(d_j, flipall( squeeze(front_a(:,:,ii,:)) ), "valid")) / batch_size;
        end
        dbiases(jj, 1) = sum(d_j(:)) / batch_size;
    end
    dweights = dweights(1:layer.rate:end,1:layer.rate:end,:,:);
end