function result = insert_zeros_into_array(array, rate)
    if rate == 1
        result = array;
    else
        [height, width, channel, batch_size] = size(array);
        result = zeros(height+(height-1)*(rate-1), width+(width-1)*(rate-1), channel, batch_size);
        result(1:rate:end, 1:rate:end, :, :) = array;
    end
end