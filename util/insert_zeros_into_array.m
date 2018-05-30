function result = insert_zeros_into_array(array, stride)
    if stride <= 1
        error('stride should > 1');
    end
    [batch_size, height, width, channel] = size(array);
    result = zeros(batch_size, height+(height-1)*(stride-1), width+(width-1)*(stride-1), channel);
    result(:, 1:stride:end, 1:stride:end, :) = array;
end