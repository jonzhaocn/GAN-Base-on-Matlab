% padding become [height+ph*2, width+p2*2, channel, batch_size]
function result = padding_height_width_in_array(array, padding_shape)
    [height, width, channel, batch_size] = size(array);
    p_top = padding_shape(1);
    p_bottom = padding_shape(2);
    p_left = padding_shape(3);
    p_right = padding_shape(4);
    result = zeros(height+p_top+p_bottom, width+p_left+p_right, channel, batch_size);
    result(p_top+1:p_top+height, p_left+1:p_left+width, :, :) = array;
end