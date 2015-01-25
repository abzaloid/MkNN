function [assigned_classes] = ownknnclassify(sample_data, training_data, group, k)

    distance_metric = 'euclidean';	% by default (then we can make it variable)

	
    % data normalization
    training_data = normalizeData(training_data);	     
    sample_data = normalizeData(sample_data);		

    C = max(group(:));	% total # of classes


    KDTree = KDTreeSearcher(training_data, 'Distance', distance_metric);
    samples_size = size(sample_data, 1);        % sample data size
    assigned_classes = zeros(samples_size, 1);  % will show at the end the corresponding classes for each sample
    for ind_sample = 1 : samples_size;
        [points_id, ~] = knnsearch(KDTree, sample_data(ind_sample, :), 'k', k);
        count_of_classes = zeros(C, 1);
        for ind_points = 1 : size(points_id, 2);
            current_point = points_id(ind_points);
            count_of_classes(group(current_point)) = count_of_classes(group(current_point)) + 1;
        end
        [~, assigned_classes(ind_sample)] = max(count_of_classes(:));
    end

end
