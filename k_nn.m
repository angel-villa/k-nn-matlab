% Angel Villa 

clc; clear;

train_file = "breast-cancer-wisconsin.csv";

% importing data
train_data = importdata(train_file);
x_train = train_data(:,1:end-1);
y_train = train_data(:,end);

k_nn = 100;

% k_val: # of validation subsets
% k_val = N -> leave-one-out
% k_val = 10 is common for many applications
k_val = 699;

error = zeros(k_val, k_nn);
for val_ind = 1:k_val
    % split data into validation training and testing sets
    val_start = ceil((val_ind-1)*size(x_train)/k_val + 0.001);
    val_end = floor(val_ind*size(x_train)/k_val);
    
    x = x_train;
    y = y_train;
    
    x(val_start:val_end,:) = [];
    y(val_start:val_end,:) = [];
    
    % split data in (x,y), (x_test, y_test) = training set, testing set
    x_test = x_train(val_start:val_end,:);
    y_test = y_train(val_start:val_end,1);
    
    N_test = size(x_test,1);
    
    N = size(x,1);
    D = size(x,2);
    
    % normalizing data
    m = zeros(D,1);
    dev = zeros(D,1);

    for i=1:D
        m(i) = mean(x(:,i));
        dev(i) = std(x(:,i));

        for j=1:N
            x(j,i) = (x(j,i) - m(i))/dev(i);
        end

        for j=1:N_test
            x_test(j,i) = (x_test(j,i) - m(i))/dev(i);
        end
    end

    % testing
    % e_i = running index for testing error
    e_i = 1;
    for k = 1:k_nn
        correct = 0;
        for n=1:N_test
            c = predict_class(x_test(n,:), k, x, y);
            if c == y_test(n)
                correct = correct + 1;
            end
        end
        Error(e_i,1) = k;
        Error(e_i,2) = (N_test - correct)/N_test;
        e_i = e_i + 1;
        error(val_ind, k) = (N_test - correct)/N_test;
    end
end

ep = zeros(1,k_nn);
for i=1:k_nn
    ep(i) = mean(error(:,i));
end

plot(1:k_nn,ep)
xlabel("k value")
ylabel("Error")
title("Error rate vs. k value Leave-one-out cross-validation k-nn")

% Predicts class of test vector x_t given k
% Computes euclidean distance from all training vectors and classifies as
% that class which appears most in the k nearest neighbors
function c = predict_class(x_t, k, x, y)
    N = size(x,1);
    nearest = [Inf(k,1) zeros(k,1)];
    % iterate through training vectors
    for i=1:N
        d = norm(x_t - x(i,:));
        % max_dist is the max euclidean distance in the current set of nn
        % ind corresponds to that index
        [max_dist, ind] = max(nearest(:,1));
        % if a smaller distance d is found than the max, replace the max 
        % with that distance d
        if d < max_dist
            nearest(ind,1) = d;
            nearest(ind,2) = y(i);
        end
    end
    c = mode_rand(nearest(:,2));
end

% Calculates mode of a vector x
% If more than one mode, selects randomly between them (MATLAB 'mode'
% function chooses smallest)
function mode_rand = mode_rand(x)
    candidates = [];
    max_count = 0;
    for i=0:9
        n = sum(x(:) == i);
        if (n > max_count)
            candidates = [i];
            max_count = n;
        elseif (n >= max_count) && ~(ismember(i,candidates))
            candidates = [candidates i];
        end
    end
    max_ind = randi(length(candidates));
    mode_rand = candidates(max_ind);
end