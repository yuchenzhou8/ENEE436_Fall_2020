%% ENEE436 Foundation of Machine Learning Project1 
% Yuchen Zhou 
% University of Maryland, College Park
% Fall 2020, 10/31/2020

%%
load('data/project1_sample_info.mat');
% sample featues
X = image_train;
% corresponding classes
Y = label_train;
%% Traing Kth-Nearest-Neighbors Model
KNN_model_1 = fitcknn(X,Y,'NumNeighbors',1,'distance','euclidean');
KNN_model_5 = fitcknn(X,Y,'NumNeighbors',5,'distance','euclidean');
KNN_model_10 = fitcknn(X,Y,'NumNeighbors',10,'distance','euclidean');
KNN_model_20 = fitcknn(X,Y,'NumNeighbors',20,'distance','euclidean');
KNN_model_50 = fitcknn(X,Y,'NumNeighbors',50,'distance','euclidean');
KNN_model_100 = fitcknn(X,Y,'NumNeighbors',100,'distance','euclidean');


%%
% run through all samples to compute the error
estimated_label = zeros(1,num_test_samples);

tic
    for n = 1:num_test_samples
        estimated_label(n) = predict(KNN_model_1, image_test(n,:));
    end
toc
%% Estimation Testing Data using training samples
% estimate test samples

estimated_label_5 = zeros(1,num_test_samples);
estimated_label_10 = zeros(1,num_test_samples);
estimated_label_20 = zeros(1,num_test_samples);
estimated_label_50 = zeros(1,num_test_samples);
estimated_label_100 = zeros(1,num_test_samples);
tic
    for n = 1:num_test_samples
        estimated_label_5(n) = predict(KNN_model_5, image_test(n,:));
        estimated_label_10(n) = predict(KNN_model_10, image_test(n,:));
        estimated_label_20(n) = predict(KNN_model_20, image_test(n,:));
        estimated_label_50(n) = predict(KNN_model_50, image_test(n,:));
        estimated_label_100(n) = predict(KNN_model_100, image_test(n,:));
    end
toc

%% Percent_Match Calculation
match_k = cell(1,6);
percent_match = zeros(1,6);
K = [1 5 10 20 50 100];

match_k{1} = (estimated_label' == label_test); % k = 1
match_k{2} = (estimated_label_5' == label_test); % k = 5
match_k{3} = (estimated_label_10' == label_test); % k = 10
match_k{4} = (estimated_label_20' == label_test); % k = 20
match_k{5} = (estimated_label_50' == label_test); % k = 50
match_k{6} = (estimated_label_100' == label_test); % k = 100
for n = 1:6 
    percent_match(n) = sum(match_k{n})/num_test_samples;
end


%% Estimate Training Data
tic
estimated_label_1_train = predict(KNN_model_1, image_train);
estimated_label_5_train = predict(KNN_model_5, image_train);
estimated_label_10_train = predict(KNN_model_10, image_train);
estimated_label_20_train = predict(KNN_model_20, image_train);
estimated_label_50_train = predict(KNN_model_50, image_train);
estimated_label_100_train = predict(KNN_model_100, image_train);
toc

%% Compute Percent Match for Training Data
match_k_train = cell(1,6);
percent_match_train = zeros(1,6);
K = [1 5 10 20 50 100];

match_k_train{1} = sum(estimated_label_1_train == label_train); % k = 1
match_k_train{2} = sum(estimated_label_5_train == label_train); % k = 5
match_k_train{3} = sum(estimated_label_10_train == label_train); % k = 10
match_k_train{4} = sum(estimated_label_20_train == label_train); % k = 20
match_k_train{5} = sum(estimated_label_50_train == label_train); % k = 50
match_k_train{6} = sum(estimated_label_100_train == label_train); % k = 100
for n = 1:6 
    percent_match_train(n) = match_k_train{n}/num_train_samples;
end


