% padding => [batch_size, height+ph*2, width+p2*2, channel]
function result = padding_height_width_in_array(array, p_top, p_bottom, p_left, p_right)
    [batch_size, height, width, channel] = size(array);
    result = zeros(batch_size, height+p_top+p_bottom, width+p_left+p_right, channel);
    result(:,p_top+1:p_top+height, p_left+1:p_left+width, :) = array;
end