prepareDataset

N = 100;
showGraphs = false;
numHiddenNodes = 50;
trainingData = cell( N, 2 );
testData = cell( N, 1 );
numSigns = size( SignSet, 1 );
models = cell( N, 1 );
testIdx = randperm( numSigns, N );
%testIdx = 1:10;

% Prepare the N sign samples
for i = 1 : N
    idx = testIdx(i);
    currentData = SignSet(idx, :);
    trainingData(i, :) = currentData(1:2);
    testData(i) = currentData(3);
    models{i} = AJDHMM( trainingData(i, :), numHiddenNodes );
    %plotTrajectory( trainingData{i, 1}, i, 'b' );
    %plotTrajectory( trainingData{i, 2}, i, 'c' );
end

% Train the N models
for i = 1 : N
    models{i}.train();
    %bt = models{i}.generateBestSequence(20);
    %plotTrajectory( bt, i, 'k' );
end

% Test
correct = 0;
incorrect = 0;
for i = 1 : N
    maxScore = -intmax;
    signIdx = 0;
    fprintf( 'Testing model %d\n', i );
    for j = 1 : N
        tScore = models{i}.computeLikelihood( testData{j} );
        fprintf( 'Example %d likelihood %f\n', j, tScore );
        if tScore > maxScore
            maxScore = tScore;
            signIdx = testIdx(j);
        end
    end
    if signIdx == testIdx(i)
        correct = correct + 1;
    else
        incorrect = incorrect + 1;
    end
end

fprintf( 'Accuracy %f\n', correct / N );
