% AJDHMM.m
% Author: Alex Dillhoff
% Date:   12/09/2015

% classdef AJDHMM
% Defines a HMM builder
classdef AJDHMM < handle
    properties
        Data
        Targets
        Criterion
        Pi          % Initial state distribution
        A           % State transition probabilities
        B           % Observation probabilities
        Sigma
        Mu
        NumStates
    end

    methods
        % Default constructor
        function obj = AJDHMM( data, numStates )
            obj.NumStates = numStates;
            obj.Criterion = 1e-5;
            numFeatures = size( data{1}, 2 );
            obj.Data = data;
            obj.Pi = obj.normalize( rand( 1, obj.NumStates ), 2 );
            obj.A  = rand( obj.NumStates );
            obj.A  = bsxfun( @rdivide, obj.A, sum( obj.A, numFeatures ) );
            obj.Mu = rand( numFeatures, obj.NumStates );
            obj.Sigma = zeros( numFeatures, numFeatures, obj.NumStates );

            % Random init based on gaussian_init in Kevin Murphy's toolbox.
            % Assign each mu to a different data point
            sample = data{1};
            C = cov( sample )
            obj.Sigma(:, :, 1) = diag( diag( C ) ) * 0.5
            idxs = randperm( size( data{1}, 1 ) );
            obj.Mu(:, 1) = sample(idxs(1), :);
        end

        % Generates a new random sequence using a learned model.
        function [O, Q] = generateRandomSequence( obj, T )
            % Choose initial state according to Pi
            numFeatures = size( obj.Data{1}, 2 );
            O = zeros( T, numFeatures );
            Q = zeros( T, 1 );

            Q(1) = sum( rand >= cumsum( obj.Pi ) ) + 1;
            S = obj.Sigma(:, :, Q(1));
            [~, p] = chol( S );
            % TODO: Figure out if this is a good idea.
            if p ~= 0
                S = S + eye( size( S, 1 ) );
            end
            O(1, :) = mvnrnd( obj.Mu(:, Q(1)), S );

            for t = 2 : T
                % Transition to a new state using A
                Q(t) = sum( rand > cumsum( obj.A(Q(t - 1), :) ) ) + 1;
                S = obj.Sigma(:, :, Q(t));
                [~, p] = chol( S );
                % TODO: Figure out if this is a good idea.
                if p ~= 0
                    S = S + eye( size( S, 1 ) );
                end

                % Choose O_t = v_k via b_i(k)
                O(t, :) = mvnrnd( obj.Mu(:, Q(t)), S );
            end
        end
        
        % Generates a max likelihood sequence using a learned model.
        function [O, Q] = generateBestSequence( obj, T )
            % Choose initial state according to Pi
            numFeatures = size( obj.Data{1}, 2 );
            O = zeros( T, numFeatures );
            Q = zeros( T, 1 );

            [~, Q(1)] = max( obj.Pi );
            S = obj.Sigma(:, :, Q(1));
            [~, p] = chol( S );
            % TODO: Figure out if this is a good idea.
            if p ~= 0
                S = S + eye( size( S, 1 ) );
            end
            O(1, :) = mvnrnd( obj.Mu(:, Q(1)), S );

            for t = 2 : T
                % Transition to a new state using A
                [~, Q(t)] = max( obj.A(:, Q(t - 1)) );

                S = obj.Sigma(:, :, Q(t));
                [~, p] = chol( S );
                % TODO: Figure out if this is a good idea.
                if p ~= 0
                    S = S + eye( size( S, 1 ) );
                end

                % Choose O_t = v_k via b_i(k)
                O(t, :) = mvnrnd( obj.Mu(:, Q(t)), S );
            end
        end

        % function ll = computeLikelihood( data )
        % Computes the likelihood of the given data using the current model
        % parameters.
        function ll = computeLikelihood( obj, data )
            B = obj.buildEmissionModel( data, obj.Mu, obj.Sigma );
            [~, ~, ~, ~, ll] = obj.forwardBackward( B );
        end

        function debugGaussian( obj )
            X = 1:20;
            X = X./20;
            Y = sin(X);
            obj.NumStates = 5;
            obj.Data = cell( 1, 1 );
            obj.Data{1} = [X; Y]'
            obj.Pi = [0.2, 0.2, 0.2, 0.2, 0.2];
            obj.A  = rand( obj.NumStates );
            obj.A  = bsxfun( @rdivide, obj.A, sum( obj.A, 2 ) );
            obj.Mu = rand( 2, obj.NumStates );
            obj.Sigma = zeros( 2, 2, obj.NumStates );
            for i = 1 : obj.NumStates
                b = rand( 2 );
                b = b * b';
                obj.Sigma(:, :, i) = b;
            end

            obj.train();

            obj.generateBestSequence( 10 )
            obj.generateRandomSequence( 10 )
        end

        % function result = normalize( arr, dimension )
        % Normalizes the given array so that all values along the given
        % dimension add up to 1.
        function result = normalize( obj, arr, dimension )
            denom = sum( arr, dimension );
            result = bsxfun( @rdivide, arr, denom );
        end

        % function buildEmissionModel()
        function B = buildEmissionModel( obj, data, mu, sigma )
            T = size( data, 1 );
            B = zeros( obj.NumStates, T );
            for j = 1 : obj.NumStates
                M = mu(:, j);
                M(:);
                M = M * ones( 1, T );
                S = squeeze( sigma(:, :, j) );
                [~, p] = chol( S );
                % TODO: Figure out if this is a good idea.
                if p ~= 0
                    S = S + eye( size( S, 1 ) );
                end

                % NOTE: Using gaussian_prob from Kevin Murphy's toolbox.
                B(j, :) = gaussian_prob( data', mu(:, j), S );
            end
        end

        % function result = checkConvergence( ll, pll, criterion )
        function result = checkConvergence( obj, ll, pll, criterion )
            result = false;
            
            if ll - pll < -1e-3
                return;
            end

            delta = abs( ll - pll );
            avg = (abs( ll ) + abs( pll ) + eps) / 2;
            if (delta / avg) < criterion
                result = 1;
            end
        end

        % function [gam, xi, ll] = forwardBackward()
        % Basic implementation of the forward backward algorithm.
        function [alpha, veta, gam, xi, ll] = forwardBackward( obj, B )
            %%%%%%%%%%%%%%
            %% FORWARDS %%
            %%%%%%%%%%%%%%
            
            numSamples = size( B, 2 );
            ll = 0;
            scale = zeros( numSamples, 1 );

            % Initialize Alpha
            alpha = zeros( obj.NumStates, numSamples );
            % NOTE: You previous erroneously calculated pi * B instead of
            % pi .* B
            alpha(:, 1) = obj.Pi' .* B(:, 1);
            scale(1) = sum( alpha(:, 1) );
            alpha(:, 1) = alpha(:, 1) / scale(1);

            % Update Alpha for each t
            for t = 2 : numSamples
                alpha(:, t) = (alpha(:, t - 1)' * obj.A)' .* B(:, t);
                scale(t) = sum( alpha(:, t) );
                alpha(:, t) = alpha(:, t) / scale(t);
            end

            ll = sum( log( scale ) );

            % Initialize veta
            veta = zeros( obj.NumStates, numSamples );
            veta(:, numSamples) = 1;

            %%%%%%%%%%%%%%%
            %% BACKWARDS %%
            %%%%%%%%%%%%%%%

            % Update veta for each t
            for t = numSamples : -1 : 2
                veta(:, t - 1) = obj.A * (B(:, t) .* ...
                    veta(:, t));
                veta(:, t - 1) = obj.normalize( veta(:, t - 1), 1 );
            end

            % Initialize Xi
            xi = zeros( obj.NumStates, obj.NumStates, numSamples - 1 );
            for t = 1 : numSamples - 1
                denom = (alpha(:, t)' * obj.A) .* B(:, t + 1)' * ...
                    veta(:, t + 1);
                for i = 1 : obj.NumStates
                    numer = alpha(i, t) .* obj.A(i, :) .* ...
                        B(:, t + 1)' .* veta(:, t + 1)';
                    xi(i, :, t) = numer / denom;
                end
            end

            % Initialize Gamma
            % P(q_t = S_i | O, lambda)
            gam = zeros( obj.NumStates, numSamples );
            gamT = alpha(:, numSamples) .* veta(:, numSamples);
            gamT = gamT / sum( gamT );
            gam(:, end) = gamT;
            for t = numSamples - 1 : -1 : 1
                gam(:, t) = obj.normalize( alpha(:, t) .* veta(:, t), 1 );
            end

        end

        % function train()
        % Trains the HMM using the current Data.
        function train( obj )
            numIterations = 1000;
            done = false;
            idx = 0;
            likelihood = -Inf;
            numFeatures = size( obj.Data{1}, 2 );
            while ~done && idx < numIterations
                tempPi = zeros( size( obj.Pi ) );
                tempA  = zeros( obj.NumStates );
                tempMu = zeros( size( obj.Mu ) );
                tempSum = zeros( numFeatures, numFeatures, obj.NumStates );
                denom = 0;
                currentLikelihood = 0;

                %%%%%%%%%%
                % E-STEP %
                %%%%%%%%%%
                
                % NOTE: DEBUGGING VARS
                %expt = zeros( obj.NumStates );
                %expv = zeros( obj.NumStates, 1 );
                %m = zeros( size( obj.Mu ) );
                %op = zeros( size( obj.Sigma ) );

                for j = 1 : size( obj.Data, 1 )
                    obs = obj.Data{j};
                    % Build Emission model
                    B = obj.buildEmissionModel( obs, obj.Mu, obj.Sigma );

                    [alpha, veta, gam, xi, ll] = obj.forwardBackward( B );
                    %[alp2, bet2, gam2, ll2, xi2] = fwdback( obj.Pi, obj.A, B );

                    %disp( ll )
                    %disp( ll2 )
                    %assert( isequal( ll, ll2 ) )

                    currentLikelihood = currentLikelihood + ll;
                    
                    tempPi = tempPi + gam(:, 1)';
                    tempA = tempA + sum( xi, 3 );

                    % NOTE: DEBUGGING
                    %expt = expt + xi2;
                    %expv = expv + gam2(:, 1);

                    % Updates for Gaussian parameters
                    denom = denom + sum( gam, 2 );

                    for i = 1 : obj.NumStates
                        temp = obs' .* repmat( gam(i, :), [numFeatures, 1] );
                        tempMu(:, i) = tempMu(:, i) + sum( temp, 2 );
                        tempSum(:, :, i) = tempSum(:, :, i) + temp * obs;
                    end
                end

                %%%%%%%%%%
                % M-STEP %
                %%%%%%%%%%

                denom = denom + (denom == 0);

                sigmaPrior = repmat( 0.01 * eye( numFeatures ), [1 1 obj.NumStates] );
                for q = 1 : obj.NumStates
                    obj.Mu(:, q) = tempMu(:, q) / denom(q);
                    obj.Sigma(:, :, q) = tempSum(:, :, q) / denom(q) - ...
                        obj.Mu(:, q) * obj.Mu(:, q)';
                end

                obj.Sigma = obj.Sigma + sigmaPrior;

                tempPi = obj.normalize( tempPi, 2 );
                tempA  = obj.normalize( tempA, 2 );
                idx = idx + 1;
                likelihood = [likelihood currentLikelihood];
                fprintf( 'Iteration %d likelihood %f\n', idx, currentLikelihood );

                done = obj.checkConvergence( currentLikelihood, ...
                    likelihood(end - 1), obj.Criterion );

                obj.Pi = tempPi;
                obj.A  = tempA;
            end
        end
    end
end
