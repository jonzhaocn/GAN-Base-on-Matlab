function layer = batch_norm(input, layer)
    if ndims(input)==4
        
    elseif ndims(input)==2
        % if ndims input == 2, reshape input into a 4-dims array
        input_shape = size(input);
        input = reshape(input, input_shape(1), 1, 1, input_shape(2));
    else 
        error('wrong input shape')
    end
    input_maps = size(input, 3);
    % ---
    mus = squeeze(mean(mean(mean(input,4),1),2));
    % ---
    temp = input;
    for i=1:input_maps
        temp(:,:,i,:) = temp(:,:,i,:) - mus(i);
    end
    temp = temp .^ 2;
    sigma2s = squeeze(mean(mean(mean(temp,4),1),2));
    
    output = zeros(size(input));
    input_hat = zeros(size(input));
    % ----
    for i=1:input_maps
        input_hat(:,:,i,:) = (input(:,:,i,:) - mus(i)) / sqrt(sigma2s(i) + layer.epslion);
        output(:,:,i,:) = layer.weights(i) * input_hat(:,:,i,:) + layer.biases(i, 1);
    end
    % ----
    layer.input = squeeze(input);
    layer.input_hat = squeeze(input_hat);
    layer.mus = mus;
    layer.sigma2s = sigma2s;
    layer.z = squeeze(output);
    % ----
end