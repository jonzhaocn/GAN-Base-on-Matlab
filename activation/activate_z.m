function a = activate_z(layer)
    a = layer.z;
    if isfield(layer, 'activation')
        if strcmp(layer.activation, 'sigmoid')
            a = sigmoid(a);
        elseif strcmp(layer.activation, 'relu')
            a = relu(a);
        elseif strcmp(layer.activation, 'leaky_relu')
            a = leaky_relu(a);
        elseif strcmp(layer.activation, 'tanh')
            a = tanh(a);
        else
            error('wrong activation function in layer')
        end
    end
end