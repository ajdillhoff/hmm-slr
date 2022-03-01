function plotTrajectory( data, figureIdx, lineOption )
    figure( figureIdx );
    plot( data(1, 1), data(1, 2), 'ro', 'MarkerSize', 12 );
    hold on;
    plot( data(:, 1), data(:, 2), lineOption, 'MarkerSize', 12 );
    plot( data(end, 1), data(end, 2), 'bo', 'MarkerSize', 12 );
end
