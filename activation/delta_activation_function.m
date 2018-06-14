function result = delta_activation_function(layer)
    result = 1;
    if isfield(layer, 'activation')
        switch layer.activation
            case 'sigmoid'
                result = delta_sigmoid(layer.z);
            case 'relu'
                result = delta_relu(layer.z);
            case 'leaky_relu'
                result = delta_leaky_relu(layer.z);
            case 'tanh'
                result = delta_tanh(layer.z);
            otherwise
                error('wrong activation function in layer:%s', layer.activation)
        end
    end
end