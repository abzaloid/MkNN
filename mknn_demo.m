{\rtf1\ansi\deff0{\fonttbl{\f0\fnil\fcharset0 Courier New;}}
{\*\generator Msftedit 5.41.21.2510;}\viewkind4\uc1\pard\lang1033\f0\fs22 % mknn demo function\par
%\par
% mknn_demo(TEXTFILE, PERCENTS, KFOLD, KS, BLOCKS, STEPSIZE)\par
%\par
% TEXTFILE is the file of data where\par
% data must have the following structure:\par
%\par
%   class_number_1, feature_1_1, feature_1_2, ..., feature_1_d\par
%   class_number_2, feature_2_1, feature_2_2, ..., feature_2_d\par
%   ...\par
%   class_number_N, feature_N_1, feature_N_2, ..., feature_N_d\par
%\par
% where,\par
%   class_number is integer number from 1 to maximum number of classes;\par
%   feature      is double number  \par
%\par
% PERCENTS is a vector of percents to reduce data\par
%\par
% KFOLD is a value of k for k-fold cross-validation\par
%\par
% KS is a vector of k's to be used in knn classification\par
%\par
% BLOCKS is a vector of block sizes to be used in mknn\par
%\par
% STEPSIZE is a value of iterations to get M percent\par
% \par
% example of using:\par
% mknn_demo('wine.txt', [1 70], 5, [1 4 10], [10 40 60 100], 3);\par
\par
function mknn_demo(textfile, percents, kfold, ks, blocks, stepsize)\par
\par
% open data file\par
data = csvread(textfile);\par
\par
data_size = size(data, 1);                      % number of total instances \par
class_size = max(data(:, 1));                    % maximum class number\par
frequency_of_class = zeros(class_size, 1);         % frequency (number of instances) of this class in data set    \par
num_of_folds = kfold;                                % number of folds\par
window_size= round(data_size / numOfFolds);     % window size which is used in k-fold validation\par
\par
% counting the frequency of each class\par
for ind = 1 : data_size;\par
    frequency_of_class(data(ind, 1)) = frequency_of_class(data(ind, 1)) + 1;\par
end\par
\par
total_accuracy= zeros(size(blocks, 2), size(percents, 2), size(ks, 2));\tab % total_accuracy(ind_block, ind_percent, ind_k)\par
total_time = zeros(size(blocks, 2), size(percents, 2), size(ks, 2));\tab % total_time(ind_block, ind_percent, ind_k) is time for REDUCTION process for \par
\tab\tab\tab\tab\tab\tab\tab\tab\tab\tab\tab\tab\tab % block : ind_block, percent to reduce (M) : ind_percent, k : ind_k \par
knn_total_time = zeros(size(blocks, 2), size(percents, 2), size(ks, 2)); % knn_total_time(ind_block, ind_percent, ind_k) is the time for knn classification\par
\par
for ind_block=1:size(blocks, 2)\par
    block_size = blocks(ind_block);\par
\par
\tab for ind_percent = 1 : size(percents, 2);\par
\tab\tab % specify the percent \par
\tab\tab percent = percents(ind_percent);\par
\tab\tab for ind_k = 1 : size(ks, 2);\par
\tab\tab\tab k = ks(ind_k);\par
\tab\tab\tab % random permutation of data\par
\tab\tab\tab data = data(randperm(data_size), :);\par
\tab\tab\tab % k-fold validation\par
\tab\tab\tab for ind_test = 1 : numOfFolds;\par
    \tab\tab\tab\tab window_interval_start\tab =   (ind_test - 1) * window_size+ 1;                                % window (fold) starting index\par
    \tab\tab\tab\tab window_interval_end   =   min(window_interval_start+ window_size- 1, data_size);           % window (fold) ending index\par
\tab     \tab\tab\tab training_indices      =   [1:(windowIntervalStart-1) (window_interval_end+1):data_size];    % training set indices among array data\par
\tab\tab\tab\tab group               \tab =   data(training_indices, 1);                                      % training set classes\par
\tab\tab\tab\tab training            \tab =   data(training_indices, 2 : end);                                % training set\par
    \tab\tab\tab\tab sample_data         \tab =   data(windowIntervalStart:window_interval_end, 2 : end);           % sample (test) set\par
    \tab\tab\tab\tab sample_group        \tab =   data(windowIntervalStart:window_interval_end, 1);                 % sample (test) classes \par
 \tab\par
    \tab\tab\tab\tab % mknn call\par
    \tab\tab\tab\tab [c, mknn_time, knn_time] = mknn_v3(percent, sample_data, training, group, k, block_size, stepsize, 'euclidean', 'no');                                \par
      \tab\par
\tab\tab\tab\tab % total accuracy accumulating, where sum(c == sample_group) means how\par
    \tab\tab\tab\tab % many classes of test set matched\par
\tab     \tab\tab\tab totalAccuracy(ind_block, ind_percent, ind_k) = totalAccuracy(ind_block, ind_percent, ind_k) + sum(c == sample_group);                             \par
    \tab\tab\tab\par
\tab\tab\tab\tab total_time(ind_block, ind_percent, ind_k) = total_time(ind_block, ind_percent, ind_k) + mknn_time;\par
    \tab\tab\tab\tab knn_total_time(ind_block, ind_percent, ind_k) = knn_total_time(ind_block, ind_percent, ind_k) + knn_time;\par
    \par
\tab\tab\tab end\par
\tab\tab\tab totalAccuracy(ind_block, ind_percent, ind_k) = totalAccuracy(ind_block, ind_percent, ind_k) / data_size;\par
\tab\tab end\par
\tab end\par
\pard\sl360\slmult1 end\par
% write results to file\par
\pard fid = fopen(strcat(textfile, '.benchmark'), 'w+');\par
fprintf(fid, 'total number of instances = %d, total number of dimensions = %d\\n', data_size, size(data, 2) - 1);\par
fprintf(fid, 'mkNN algorithm with 5-fold cross-validation on %s dataset:\\n', textfile);\par
\par
for ind_block = 1 : size(blocks, 2);\par
    block_size = blocks(ind_block);\par
    fprintf(fid, '\\n============================================\\n');\par
    fprintf (fid, 'for block size = %d\\n', block_size);\par
    for ind_k = 1 : size(ks,2);\par
    k = ks(ind_k);\par
    fprintf(fid, '\\n------------------------------------------------------------------------------\\n');\par
    fprintf(fid, 'for k = %d', k);\par
    fprintf(fid, '\\nM_______ = ');\par
    for ind_percent = 1 : size(percents,2);\par
        percent = percents(ind_percent);\par
        fprintf(fid, '%15.2f%%', percent);\par
    end\par
    fprintf(fid, '\\nAccuracy = ');\par
    for ind_percent = 1 : size(percents,2);\par
        fprintf(fid, '%15.2f%%', 100*totalAccuracy(ind_block, ind_percent, ind_k));\par
    end\par
    fprintf(fid, '\\nTime_____ = ');\par
    for ind_percent = 1 : size(percents,2);\par
        fprintf(fid, '%9.2fs, %.2fs', total_time(ind_block, ind_percent, ind_k), knn_total_time(ind_block, ind_percent, ind_k));\par
    end\par
    end\par
end\par
fprintf(fid, '\\n\\n');\par
\par
\par
\par
\par
\par
% test of only knn classification without reduction process\par
\par
\par
\par
total_accuracy= zeros(size(ks, 2));\par
total_time = zeros(size(ks, 2));\par
for ind_k = 1 : size(ks, 2);\par
k = ks(ind_k);\par
\par
\par
% random permutation of data\par
data = data(randperm(data_size), :);\par
\par
startTime = cputime;\par
\par
% k-fold validation\par
for ind_test = 1 : numOfFolds;\par
    \par
    window_interval_start=   (ind_test - 1) * window_size+ 1;                                % window (fold) starting index\par
    window_interval_end   =   min(window_interval_start+ window_size- 1, data_size);           % window (fold) ending index\par
    training_indices    =   [1:(windowIntervalStart-1) (window_interval_end+1):data_size];    % training set indices among array data\par
    group               =   data(training_indices, 1);                                      % training set classes\par
    training            =   data(training_indices, 2 : end);                                % training set\par
    sample_data         =   data(windowIntervalStart:window_interval_end, 2 : end);           % sample (test) set\par
    sample_group        =   data(windowIntervalStart:window_interval_end, 1);                 % sample (test) classes \par
 \par
    % knn classification of data using kd-trees\par
    c = ownknnclassify(sample_data, training, group, k);\par
    \par
    % total accuracy accumulating, where sum(c == sample_group) means how\par
    % many classes of test set matched\par
    totalAccuracy(ind_k) = totalAccuracy(ind_k) + sum(c == sample_group);                             \par
\par
end\par
totalAccuracy(ind_k) = totalAccuracy(ind_k) / data_size;\par
finishTime = cputime;\par
total_time(ind_k) = finishTime - startTime;\par
end\par
\par
fprintf(fid, 'knnclassify built-in algorithm with 5-fold cross-validation on %s dataset:\\n\\n', textfile);\par
for ind_k = 1 : size(ks,2);\par
    k = ks(ind_k);\par
    fprintf(fid, '\\n------------------------------------------------------------------------------\\n');\par
    fprintf(fid, 'for k = %d', k);\par
    fprintf(fid, '\\nAccuracy  = ');\par
    fprintf(fid, '%12.2f%%', 100*totalAccuracy(ind_k));\par
    fprintf(fid, '\\nTime      = ');\par
    fprintf(fid, '%12.2fs', total_time(ind_k));\par
 end\par
fprintf(fid, '\\n');\par
end\par
}
 
