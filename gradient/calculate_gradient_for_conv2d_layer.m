function [dweights, dbiases] = calculate_gradient_for_conv2d_layer(front_a, layer)
    d = layer.d;
    weights = layer.weights;
    dweights = zeros(size(weights));
    dbiases = zeros(size(layer.biases));
    batch_size = size(d, 4);
    % padding
    d = padding_height_width_in_array(d, layer.padding_shape);
    % front_a [height, width, in_channel, batch_size]
    % d [height, width, out_channel, batch_size]
    for jj = 1:size(weights,4) %output channel
        d_j = squeeze(d(:,:,jj,:));
        for ii=1:size(weights,3) % input channel
            dweights(:,:,ii,jj) =  squeeze(convn(d_j, flipall( squeeze(front_a(:,:,ii,:)) ), "valid")) / batch_size;
        end
        dbiases(1,jj) = sum(d_j(:)) / batch_size;
    end
end