function options = argparse(options, args)
    params = fieldnames(args);
    for i = 1:numel(params)
        p = char(params(i));
        if isfield(options, p)
            options.(p) = args.(p);
        else
            error('xxx')
        end
    end
end