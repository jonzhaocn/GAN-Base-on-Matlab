% get error term from batch norm layer with respect to current layer's
% output
function result = get_error_term_from_batch_norm_layer(back_layer)
    d = back_layer.d;
    mus = back_layer.mus;
    sigma2s = back_layer.sigma2s;
    input_shape = back_layer.input_shape;
    input = back_layer.input;
    if numel(input_shape) == 4
        
    elseif numel(input_shape) == 2
        input = reshape(input, input_shape(1), 1, 1, input_shape(2));
        input_shape = size(input);
    else
        error('error input shape')
    end
    input_maps = input_shape(3);
    der_x_hat = zeros(input_shape);
    der_x = zeros(input_shape);
    der_sigma2s = zeros(input_maps);
    der_mu = zeros(input_maps);
    for i = 1:input_shape(3)
        m = numel(d(:,:,i,:));
        der_x_hat(:,:,i,:) = d(:,:,i,:) * back_layer.weights(i);
        temp = der_x_hat(:,:,i,:) .* (input(:,:,i,:) - mus(i)) * (-0.5) * (sigma2s(i) + back_layer.epslion)^(-1.5);
        der_sigma2s(i) = sum(temp(:));
        temp = der_x_hat(:,:,i,:) * (-1/sqrt(sigma2s(i)+back_layer.epslion));
        temp2 = -2 * (input(:,:,i,:)-mus(i));
        temp2 = der_sigma2s(i) * sum(temp2(:)) * (1/m);
        der_mu(i) = sum(temp(:)) + temp2;
        der_x(:,:,i,:) = der_x_hat(:,:,i,:) * (1/sqrt(sigma2s(i) + back_layer.epslion)) + ...
            der_sigma2s(i) * 2 * (input(:,:,i,:)-mus(i)) * (1/m) + ...
            der_mu(i) * (1/m);
    end
    result = squeeze(der_x);
end