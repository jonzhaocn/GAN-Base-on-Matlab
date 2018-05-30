function result = insert_zeros_into_filter(filter, rate)
    [filter_height, filter_width, filter_in_channel, out_channel] = size(filter);
    if rate == 1
        result = filter;
    elseif rate > 1
        result = zeros(filter_height+(filter_height-1)*(rate-1), filter_width+(filter_width-1)*(rate-1), filter_in_channel, out_channel);
        result(1:rate:end, 1:rate:end, :, :) = filter;
    end
end