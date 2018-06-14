function [dweights, dbiases] = calculate_gradient_for_fully_connect_layer(front_a, layer)
    dweights = front_a * layer.d' / size(layer.d, 2);
    dbiases = mean(layer.d, 2);
end