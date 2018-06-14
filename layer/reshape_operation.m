function result = reshape_operation(array, output_shape)
    if numel(array) == prod(output_shape)
        result = reshape(array, output_shape);
    else
        input_shape = size(array);
        count = input_shape(end);
        result = reshape(array, [output_shape(1:end-1), count]);
    end
end