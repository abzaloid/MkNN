function normalizedData = normalizeData (data)
    n = size(data, 1);
    max_values = max(data(:, :));
    min_values = min(data(:, :));
    normalizedData = (data(:, :) - repmat(min_values, n, 1)) ./ (repmat(max_values-min_values, n, 1));
end
