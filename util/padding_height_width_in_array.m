% padding become [height+ph*2, width+p2*2, channel, batch_size]
function result = padding_height_width_in_array(array, p_top, p_bottom, p_left, p_right)
    [height, width, channel, batch_size] = size(array);
    result = zeros(height+p_top+p_bottom, width+p_left+p_right, channel, batch_size);
    result(p_top+1:p_top+height, p_left+1:p_left+width, :, :) = array;
end