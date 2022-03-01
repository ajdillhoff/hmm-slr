load handface_data_combined.mat;

numSigns = size( sg1, 1 );
SignSet = cell( numSigns, 3 );
for i = 1 : numSigns
    lh = sg1{i, 1};
    %rh = sg1{i, 2};
    obs = [];
    %if isempty( rh )
        obs = lh(:, 1:2);
    %else
        %obs = lh(:, 1:2);
        %rh = rh(:, 1:2);
        %if size( rh, 1 ) ~= size( lh, 1 )
            %rh = rh(1:size( lh, 1 ), :);
        %end
        %obs = [obs rh];
    %end
    SignSet{i, 1} = obs;

    % sample 2
    lh = sg2{i, 1};
    %rh = sg2{i, 2};
    %if isempty( rh )
        %obs = lh(:, 1:2);
    %else
        obs = lh(:, 1:2);
        %rh = rh(:, 1:2);
        %if size( rh, 1 ) ~= size( lh, 1 )
            %rh = rh(1:size( lh, 1 ), :);
        %end
        %obs = [obs rh];
    %end
    SignSet{i, 2} = obs;
    
    % sample 3
    lh = sg3{i, 1};
    %rh = sg3{i, 2};
    %if isempty( rh )
        %obs = lh(:, 1:2);
    %else
        obs = lh(:, 1:2);
        %rh = rh(:, 1:2);
        %if size( rh, 1 ) ~= size( lh, 1 )
            %rh = rh(1:size( lh, 1 ), :);
        %end
        %obs = [obs rh];
    %end
    SignSet{i, 3} = obs;
    SignSet(i, :) = normalizeData( SignSet(i, :) );
end
