function [dweights, dbiases] = calculate_gradient_for_batch_norm_layer(front_a, layer)
    d = layer.d;
    input_hat = layer.input_hat;
    weights = layer.weights;
    biases = layer.biases;
    dweights = zeros(size(weights));
    dbiases = zeros(size(biases));
    if ndims(input_hat) == 4
        
    elseif ndims(input_hat) == 2
        shape = size(input_hat);
        input_hat = reshape(input_hat, shape(1), 1, 1,  shape(2));
    else
        error('xxx')
    end
    input_maps = size(front_a, 3);
    for i = 1:input_maps
        d_i = d(:,:,i,:);
        temp = d_i .* input_hat(:,:,i,:);
        dweights(i) = sum(temp(:));
        dbiases(i) = sum(d_i(i));
    end
end