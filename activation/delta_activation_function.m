function result = delta_activation_function(layer)
    result = 1;
    if isfield(layer, 'activation')
        if strcmp(layer.activation, 'sigmoid')
            result = delta_sigmoid(layer.z);
        elseif strcmp(layer.activation, 'relu')
            result = delta_relu(layer.z);
        elseif strcmp(layer.activation, 'leaky_relu')
            result = delta_leaky_relu(layer.z);
        else
            error('wrong activation function in layer')
        end
    end
end