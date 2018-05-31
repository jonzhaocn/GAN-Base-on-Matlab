function check_layer_field_names(layer, required_fields, optional_fields)
    for i = 1:numel(required_fields)
        field_name = required_fields{i};
        if ~isfield(layer, field_name)
            error([field_name, ' is required in ', layer.type])
        end
    end
    current_field_names = fieldnames(layer);
    for i = 1:numel(current_field_names)
        current_field = current_field_names{i};
        if ~ismember(current_field, [required_fields, optional_fields])
            error(['redundant filed in ', layer.type, ':', current_field])
        end
    end
end