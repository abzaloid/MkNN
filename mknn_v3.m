%  MKNN VERSION 3 
%  mknn_v3 classifies data using the nearest-neighbor method and can
%  visualize the data reduction process
%  
%     [CLASS, MKNN_TIME, KNN_TIME] = mknn_v3(PERCENT,SAMPLE,TRAINING,GROUP,K,BLOCKSIZE,STEPSIZE, DISTANCE,PCA) classifies each row of the
%     data in SAMPLE into one of the groups in TRAINING using the nearest-
%     neighbor method. SAMPLE and TRAINING must be matrices with the same
%     number of columns. GROUP is a grouping variable for TRAINING. Its
%     unique values define groups, and each element defines the group to
%     which the corresponding row of TRAINING belongs. GROUP must be a
%     numeric vector. TRAINING and GROUP must have the same number of rows. 
%     CLASS indicates which group each row of SAMPLE has
%     been assigned to, and is of the same type as GROUP.
%     
%	MKNN_TIME is time used for data reduction
%	KNN_TIME is the time used for reduced data classification using knn
%  
%     PCA is :
%             ['no']              means not to visualize data
%             ['PCA']             means reduce data dimension to 2
%             [ind1 ind2]         means use only ind1 and ind2 data dimensions 
%             []                  means use only 1st and 2nd data dimensions
%
%	
%     PERCENT is a real value from 0 to 100 which specifies how want we
%     decrease our training set. For example, if PERCENT is 20, then you
%     training set will be reduced up to 20%.
%
%     BLOCKSIZE is a integer value from 2 to training size, which specifies
%     the size of chunks we divide the training data and proceed MkNN
%     
%     [CLASS, MKNN_TIME, KNN_TIME] = mknn_v3(PERCENT,SAMPLE,TRAINING,GROUP,K,BLOCKSIZE, DISTANCE,PCA) allows you to specify K,
%     the number of nearest neighbors used in the classification. 
%  
%     [CLASS, MKNN_TIME, KNN_TIME] = mknn_v3(PERCENT,SAMPLE,TRAINING,GROUP,K,BLOCKSIZE, DISTANCE,PCA) allows you to
%     select the distance metric. Choices are
%               'euclidean'    Euclidean distance
%               'cityblock'    Sum of absolute differences
%     
%     STEPSIZE allows you to reduce data to M percent step by step in
%     STEPSIZE iterations
%         
%     Examples:
%  
%        training data: two normal components
%        training = [mvnrnd([ 1  1],   eye(2), 100); ...
%                    mvnrnd([-1 -1], 2*eye(2), 100)];
%        group = [ones(100,1); 2*ones(100,1)];
%        gscatter(training(:,1),training(:,2),group);hold on;
%  
%        some random sample data
%        sample = unifrnd(-5, 5, 100, 2);
%        classify the sample using the nearest neighbor classification
%        [c3, mknn_time, knn_time] = mknn_v3(90, sample, training, group, 1, 100, 'euclidean', 'no');
%  
%        gscatter(sample(:,1),sample(:,2),c,'mc'); hold on;
%        [c3, mknn_time, knn_time] = mknn_v3(90, sample, training, group, 3, 100, 'euclidean', 'no');
%        gscatter(sample(:,1),sample(:,2),c3,'mc','o');


function [assigned_classes, mknn_time, knn_time] = mknn_v3 (percent, samples, training, group, K, blocksize, step_size, distance_metric, PCA)

    % normalizing data
    training = normalizeData(training);     
    samples = normalizeData(samples);
    
    training_size = size(training, 1);          % training data size
    M = round(training_size * percent / 100.0); % the size to which we want to reduce the training data
    dim = size(training, 2);                    % dimension of training data
    
    if (dim < 2)
        error('Dimension of data must be greater than 1');
    end
    
    visualize   = true;     % boolean is true when user specified to visualize data, and false otherwise
    visualize_ind1 = 0;     % index of 1st dimension to visualize
    visualize_ind2 = 0;     % index of 2nd dimension to visualize
    use_PCA = false;        % boolean is true when user specified using PCA, and false otherwise
    
    if (ischar(PCA) == 1)
        PCA = cellstr(PCA);
        if (~strcmp(PCA, 'PCA') && ~strcmp(PCA, 'no'))
            error ('No %s function. Use PCA keyword', PCA);
        end
        if (strcmp(PCA, 'no'))
            % do not visualize data
            visualize = false;
        else
            % then use PCA
            use_PCA = true;
        end
    elseif (size(PCA) == 0)
        % then use the first dimensions
        visualize_ind1 = 1;
        visualize_ind2 = 2;
    elseif (size(PCA) == [1 2])
        visualize_ind1 = PCA(1);
        visualize_ind2 = PCA(2);
    else
        error('PCA size must be in range [0, 2]');
    end
   
    if (training_size ~= size(group, 1))                    
        error('size of training and group must be equal');
    end
    
    if (M > training_size || M < 0)                         
        error('PERCENT must be in range [0, 100]');
    end;
    
    if (K < 1 || K > training_size)
        error('K must be in range [1, N]');
    end
    
    C                =   max(group(:));  % the total number of classes
    class_frequency  =   zeros(C, 1);    % class_frequency keeps the frequency (size) for each class in training data

    if (C > M)
        error('Please use greater than %.2f value for PERCENT', percent);
    end
    
    % doubling the size of training and group arrays to increase
    % performance while adding new instances
    temp = training;
    training = zeros(2 * training_size, dim);
    training(1:training_size, :) = temp(:, :);
    temp = group;
    group = zeros(2 * training_size, 1);
    group(1:training_size) = temp(:);
    temp = [];
    

    
    
    % counts the frequency for each class in training data
    for ind = 1 : training_size;
        class_frequency(group(ind)) = class_frequency(group(ind)) + 1;
    end

    class_fixed_frequency = class_frequency;
    
    weight =  cell(C, 1);   % weight{C}(i) is the weight of ith instance of class C
    
    index_data        =   cell(C,1);  % cell where index_data{C} = Nx1 matrix, N--number of instances, 
                                      % index_data{C}(i) means the index of instance i of class C 
                                      % in matrix training it is used to easily access data
    
    index_in_distance_matrix = cell(C, 1);

    class_last_size   =   ones(C);    % the number of instance for each class
                                      % class_last_size(C) means the number of instances of class C in training data
    
    distance_matrix = cell(C, 1);     % distance_matrix{C}(ind1, ind2) is the distance between instances
                                      % ind1 and ind2 of class C
    
    class_data = cell(C, 1);          % class_data{C}(ind, 1:dim) keeps the position of instance ind of class C
    
    
    % initializing class_data, index_data, weight, distance_matrix sizes
    for ind_class = 1 : C;
        class_data{ind_class} = zeros(class_frequency(ind_class), dim);
        index_data{ind_class} = zeros(class_frequency(ind_class), 1);
        weight{ind_class} = ones(class_frequency(ind_class), 1);
    end
    
    % filling index_data, class_data and class_last_size with data
    for ind = 1 : training_size;
        current_class = group(ind);
        index_data{current_class}(class_last_size(current_class)) = class_last_size(current_class);
        class_data{current_class}(class_last_size(current_class), :) = training(ind, :);
        class_last_size(current_class) = class_last_size(current_class) + 1;
    end
    
    
    % added_size represents how many new instances are added to the
    % training array
    added_size = 0;
    
     % prepare colors for data visualization
    % takes colors distributed uniformly
    col = jet(C);

    mknn_time = cputime;	% start counting time for data reduction process
    
    % repeat until we reduce the size to M
    current_size = training_size;
    current_percent = round(M / step_size);
    step_percent = current_percent;
    last_total_size = -1;
    while (current_size > M);
        % shuffling the data
        for ind_class = 1 : C;
            shuffle_index = randperm(class_frequency(ind_class));
            index_data{ind_class}(:) = index_data{ind_class}(shuffle_index);
        end
        
        already_used_instances_count = zeros(C, 1); % already_used_instances_count(C) is the number of instances of class C which are already assigned in blocks
        
        
        while (1 > 0)   % do until all instances of all classes will be used 
          % in each class we will consider first class_block instances
            class_block = zeros(C, 1);
            current_class = 1;
            already_chosen_instances_size = 0;
            found_non_empty_class = 0;
            actual_blocksize = 0;
            while (already_chosen_instances_size < blocksize); 
                if (class_frequency(current_class) > already_used_instances_count(current_class) + class_block(current_class))
                    class_block(current_class) = class_block(current_class) + 1;
                    already_chosen_instances_size = already_chosen_instances_size + 1;
                    found_non_empty_class = 1;
                    actual_blocksize = actual_blocksize + 1;
                end
                current_class = current_class + 1;
                if (current_class > C)
                    if (~found_non_empty_class)
                        break;
                    end
                    found_non_empty_class = 0;
                    current_class = 1;
                end
            end
            if (sum(class_block(:)) < 2)
                break;
            end
            for ind_class = 1 : C;
                distance_matrix{ind_class} = zeros(class_block(ind_class));
                index_in_distance_matrix{ind_class} = 1 : class_block(ind_class);
            end
            for ind_class = 1 : C;
                for ind1 = already_used_instances_count(ind_class) + 1 : class_block(ind_class) + already_used_instances_count(ind_class);
                    for ind2 = ind1 : already_used_instances_count(ind_class) + 1 : class_block(ind_class) + already_used_instances_count(ind_class);
                        current_distance = getDistance(class_data{ind_class}(index_data{ind_class}(ind1), :), class_data{ind_class}(index_data{ind_class}(ind2), :), distance_metric);
                        distance_matrix{ind_class}(index_in_distance_matrix{ind_class}(ind1 - already_used_instances_count(ind_class)) , index_in_distance_matrix{ind_class}(ind2 - already_used_instances_count(ind_class))) = current_distance;
                        distance_matrix{ind_class}(index_in_distance_matrix{ind_class}(ind2 - already_used_instances_count(ind_class)) , index_in_distance_matrix{ind_class}(ind1 - already_used_instances_count(ind_class))) = current_distance;
                    end
                end
            end
        
            reduced_size = round(current_percent * actual_blocksize / 100);
            current_percent = min(M, current_percent + step_percent);
            last_current_size = -1;
            while (actual_blocksize > reduced_size && current_size > M); 
                min_distance = inf; % will have the minimum distance between all possible pairs
                instance1 = -1;     % will have index of instance1
                instance2 = -1;     % will have index of instance2
                found_class = -1;   % where are both instance1 and instance2 are of class found_class
            
                % find minimum distance among all possible pairs 
                % of instances ind1 and ind2 for each class
                for ind_class = 1 : C;
                    for ind1 = 1 : class_block(ind_class);
                       for ind2 = (ind1 + 1) : class_block(ind_class);
                         current_distance = distance_matrix{ind_class}(index_in_distance_matrix{ind_class}(ind1) , index_in_distance_matrix{ind_class}(ind2));
                         if (current_distance < min_distance)
                            min_distance = current_distance;
                            instance1 = ind1 + already_used_instances_count(ind_class);
                            instance2 = ind2 + already_used_instances_count(ind_class);
                            found_class = ind_class;
                         end
                       end
                    end  
                end
              % instance1 and instance2 are the closest instances
                % found_class is the class which they belong to
        
              % if such pair was not found break
                if (found_class == -1)
                    break;
                end
        
              % new weight is the sum of weights of two closest instances
                new_weight = weight{found_class}(index_data{found_class}(instance1)) + weight{found_class}(index_data{found_class}(instance2));     
    
                % new instance is the mean of two closest instances
                new_point = (weight{found_class}(index_data{found_class}(instance1)) * class_data{found_class}(index_data{found_class}(instance1),:) ... 
                            + weight{found_class}(index_data{found_class}(instance2)) * class_data{found_class}(index_data{found_class}(instance2), :)) ...
                            / new_weight;
                % removing instance1 and instance2 from index_data 
                index_data{found_class}([instance1 instance2]) = [];
                % decrease the class_frequency by 1, since we removed 2 instances
                % and added new one
                class_frequency(found_class) = class_frequency(found_class) - 1;
                class_block(found_class) = class_block(found_class) - 1;
                % adding new instance to data array class_data and weight
                added_size = added_size + 1;
                new_point_index = class_fixed_frequency(found_class) + added_size;
                index_data{found_class}(class_frequency(found_class)) = new_point_index;
                class_data{found_class}(new_point_index, :) = new_point(:);
                weight{found_class}(new_point_index) = new_weight;
                % adding new row and new coloumn to distance_matrix which is the new
                % instance
                for ind = 1 : class_block(found_class);
                    current_distance = getDistance(new_point, class_data{found_class}(index_data{found_class}(ind),:), distance_metric);
                    distance_matrix{found_class}(index_in_distance_matrix{found_class}(ind) , class_block(found_class) + added_size) = current_distance;
                    distance_matrix{found_class}(class_block(found_class) + added_size  , index_in_distance_matrix{found_class}(ind)) = current_distance;
                end
                current_size = current_size - 1;
                actual_blocksize = actual_blocksize - 1;
                % data visualization
        
                if (visualize == false)
                    % continue without visualization
                    continue;
                end
        
                if (visualize_ind1 ~= 0)
                    % use dimensions visualize_ind1 and visualize_ind2 to visualize
                    % the data
                   figure('units','normalized','outerposition',[0 0 1 1]);          % maximize figure window
                   hold on;
                    % points plotting with size depended on their weights 
                   for ind_class = 1 : C;
                    for instance_ind = 1 : class_frequency(ind_class);
                        plot (class_data{ind_class}(index_data{ind_class}(instance_ind), visualize_ind1), ...
                                class_data{ind_class}(index_data{ind_class}(instance_ind), visualize_ind2), ...
                                '.', 'MarkerSize', 5 * weight{ind_class}(index_data{ind_class}(instance_ind)), 'Color', col(ind_class,:));
                    end
                   end     
                   hold off;
                   pause
                    close all
                elseif (use_PCA == true)
                    % use PCA
                    % combine all points to apply PCA
                    temp_class_data = zeros(sum(class_frequency(:)), dim);
                    temp_size = 0;
                    for ind_class = 1 : C;
                        for instance_ind = 1 : class_frequency(ind_class);
                            temp_size = temp_size + 1;
                            temp_class_data (temp_size, :) = class_data{ind_class}(index_data{ind_class}(instance_ind),:);
                        end
                    end
            
                    % w_transform is a transformation matrix to make 2d data
                    w_transform = princomp(temp_class_data);
                    w_transform = w_transform(:, 1:2);
            
                    figure('units','normalized','outerposition',[0 0 1 1]);          % maximize figure window
                    hold on;
                    % points plotting with size depended on their weights
                    for ind_class = 1 : C;
                        for instance_ind = 1 : class_frequency(ind_class);
                            current_point = class_data{ind_class}(index_data{ind_class}(instance_ind),:);
                            new_point = w_transform' * current_point(:);
                            plot (new_point(1), new_point(2), ...
                                '.', 'MarkerSize', 5 * weight{ind_class}(index_data{ind_class}(instance_ind)), 'Color', col(ind_class,:));
                        end    
                    end     
                    hold off;
                    pause;
                    close all
                end
                % no changes happened
                if (current_size == last_current_size) 
                    break;
                end
                last_current_size = current_size;
            end
            already_used_instances_count(:) = already_used_instances_count(:) + class_block(:);
        end
        
        % if no reduction happened then break
        if (last_total_size == current_size)
            break;
        end
        last_total_size = current_size;
    end
    
    mknn_time = cputime - mknn_time;	% save used time for data reduction in mknn_time
    
   
    % simple KNN

    
    points_size = sum(class_frequency(:));   
    cur_size = 0;
    points = zeros(points_size, dim);
    weights = zeros(points_size, 1);
    classes = zeros(points_size, 1);
    for ind_class = 1 : C;
        for ind_instance = 1 : class_frequency(ind_class);
            cur_size = cur_size + 1;
            points(cur_size, :) = class_data{ind_class}(index_data{ind_class}(ind_instance),:);
            weights(cur_size) = weight{ind_class}(index_data{ind_class}(ind_instance));
            classes(cur_size) = ind_class;
        end
    end
    
    knn_time = cputime; % start counting time for knn classification of reduced data

    KDTree = KDTreeSearcher(points, 'Distance', distance_metric);
    samples_size = size(samples, 1);            % sample data size
    assigned_classes = zeros(samples_size, 1);  % will show at the end the corresponding classes for each sample
    for ind_sample = 1 : samples_size;
        [points_id, ~] = knnsearch(KDTree, samples(ind_sample, :), 'k', K);
        count_of_classes = zeros(C, 1);
        for ind_points = 1 : size(points_id, 2);
            current_point = points_id(ind_points);
            count_of_classes(classes(current_point)) = count_of_classes(classes(current_point)) + weights(current_point);
        end
        [~, assigned_classes(ind_sample)] = max(count_of_classes(:));
    end

    knn_time = cputime - knn_time;	% save time used to classify reduced data using knn to knn_time 
    
end
    
