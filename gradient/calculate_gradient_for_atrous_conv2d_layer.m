function [dfilter, dbiases] = calculate_gradient_for_atrous_conv2d_layer(front_a, layer)
    d = layer.d;
    filter = layer.filter;
    filter = insert_zeros_into_filter(filter, layer.rate);
    dfilter = zeros(size(filter));
    dbiases = zeros(size(layer.biases));
    batch_size = size(d, 4);
    % padding
    d = padding_height_width_in_array(d, layer.padding_shape);
    
    for jj = 1:size(filter,4) %output channel
        d_j = squeeze(d(:,:,jj,:));
        for ii=1:size(filter,3) % input channel
            dfilter(:,:,ii,jj) =  squeeze(convn(d_j, flipall( squeeze(front_a(:,:,ii,:)) ), "valid")) / batch_size;
        end
        dbiases(1,jj) = sum(d_j(:)) / batch_size;
    end
    dfilter = dfilter(1:layer.rate:end,1:layer.rate:end,:,:);
end