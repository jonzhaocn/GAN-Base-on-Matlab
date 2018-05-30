function a = activate_z(layer)
    a = layer.z;
    if isfield(layer, 'activation')
        if strcmp(layer.activation, 'sigmoid')
            a = sigmoid(layer.z);
        elseif strcmp(layer.activation, 'relu')
            a = relu(layer.z);
        elseif strcmp(layer.activation, 'leaky_relu')
            a = leaky_relu(layer.z);
        else
            error('wrong activation function in layer')
        end
    end
end