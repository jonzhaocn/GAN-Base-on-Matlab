function a = activate_z(layer)
    a = layer.z;
    if isfield(layer, 'activation')
        switch layer.activation
            case 'sigmoid'
                a = sigmoid(a);
            case 'relu'
                a = relu(a);
            case 'leaky_relu'
                a = leaky_relu(a);
            case 'tanh'
                a = tanh(a);
            otherwise
                error('wrong activation function in layer:%s', layer.activation)
        end
    end
end