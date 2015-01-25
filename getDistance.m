% Finds the distance between points A and B
% A and B are vectors with the same length
% currently 'euclidean', 'cityblock', 'cosine' distances are available
%
% getDistance([a1 ... an], [b1 ... bn])
% by default, eucledian distance is used
%
% getDistance([a1 ... an], [b1 ... bn], 'type of metric')
%
%   Example:
%       getDistance([0 1], [1 2])
%
%   outputs:
%       1.414
%

function len = getDistance (a, b, type)
    if (strcmp(type, 'euclidean'))
        len = sqrt(sum((a(:) - b(:)).^2));
    elseif (strcmp(type, 'cityblock'))
        len = sum(abs(a(:)-b(:)));
    elseif (strcmp(type, 'cosine'))
        len = 1 - sum(a(:).*b(:)) / sqrt(sum(a(:).^2) * sum(b(:).^2));
    else
        error('%s is not available metric\n', type);
    end
end
