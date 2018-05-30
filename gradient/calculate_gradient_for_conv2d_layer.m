function [dfilter, dbiases] = calculate_gradient_for_conv2d_layer(front_a, layer)
    % front_a [batch_size, height, width, in_channel]
    % d [batch_size, height, width, out_channel]
    d = layer.d;
    filter = layer.filter;
    dfilter = zeros(size(filter));
    dbiases = zeros(size(layer.biases));
    if strcmp(layer.padding, "valid")
        % empty
        p_top = size(filter,1)-1;
        p_left = size(filter,2)-1;
        d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
    elseif strcmp(layer.padding, "same")
        p_top = floor(size(filter,1)/2);
        p_left = floor(size(filter,2)/2);
        d = padding_height_width_in_array(d, p_top, p_top, p_left, p_left);
    end
    % after being permuted ,a and d will become [height, width, batch_size, channel]
    front_a = permute(front_a, [2,3,1,4]);
    d = permute(d, [2,3,1,4]);
    for jj = 1:size(filter,4) %output channel
        d_j = d(:,:,:,jj);
        for ii=1:size(filter,3) % input channel
            dfilter(:,:,ii,jj) =  squeeze(convn(d_j, flipall(front_a(:,:,:,ii)), "valid")) / size(d, 3);
        end
        dbiases(1,jj) = sum(d_j(:)) / size(d_j, 3);
    end
end