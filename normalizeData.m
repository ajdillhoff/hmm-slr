function result = normalizeData( data )
    % normalizeData( data )
    % Normalizes the data so that the values are between 0 and 1

    % find min
    result = data;

    for i = 1 : size( data, 2 )
        temp = result{i};

        for j = 1 : size( temp, 2 )
            minV = min( temp(:, j) );
            maxV = max( temp(:, j) );
            temp(:, j) = temp(:, j) - minV;
            temp(:, j) = temp(:, j) / (maxV - minV);
        end
        result{i} = temp;
    end
end
