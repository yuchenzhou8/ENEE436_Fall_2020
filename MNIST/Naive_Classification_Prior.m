%% ENEE436 Foundation of Machine Learning Project1 
% Yuchen Zhou 
% University of Maryland, College Park
% Fall 2020, 10/31/2020

%% Training Model
% Using ML estimation to compute mu and sigma of each pixel of each label
load('data/project1_sample_info.mat');
mu_n = cell(1,10);
sigma_n = cell(1,10);
temp = cell(1,10);
index_zeros = cell(1,10);
for n = 1:10
    mu_n{n} = mean(grouped_train_sample{n});
    temp{n} = sum((grouped_train_sample{n} - repmat(mu_n{n},size(grouped_train_sample{n},1),1)).^2);
    sigma_n{n} = sqrt(temp{n}./(size(grouped_train_sample{n},1)));
end

%% Prior Probabiltiy
Prior_P = zeros(1,10);
for n = 1:10
    Prior_P(n) = size(grouped_train_sample{n},1)/num_train_samples;
end

%% Find which pixels has 0 variance
index_sum  = zeros(1,784);
for n = 1:10
     index_sum = index_sum + (sigma_n{n} ~= 0);
end

%%
for n = 1: 10
    mu_n{n} = mu_n{n}(:,index_sum == 10);
    sigma_n{n} = sigma_n{n}(:,index_sum == 10);
    grouped_train_sample{n} = grouped_train_sample{n}(:,index_sum == 10);
    grouped_test_sample{n} = grouped_test_sample{n}(:,index_sum == 10);
end

%%
tic
% run through all samples to compute the error
count_train_correct = zeros(1,10);
count_test_correct = zeros(1,10);
percent_error_train = zeros(1,10);
percent_error_test = zeros(1,10);
% number of classes (labels)
for n = 1: 10  
% number of training samples in each class 
    for m = 1: size(grouped_train_sample{n},1)
        if (n - 1) ==  Naive_Bayes_Classification(mu_n,sigma_n,grouped_train_sample{n}(m,:),Prior_P)
            count_train_correct(n) = count_train_correct(n)+1;
        end
    end
% number of testing samples in each class     
    for o = 1: size(grouped_test_sample{n},1)
        if (n - 1) ==  Naive_Bayes_Classification(mu_n,sigma_n,grouped_test_sample{n}(o,:),Prior_P)
            count_test_correct(n) = count_test_correct(n)+1;
        end
    end
    
    percent_error_train(n) = count_train_correct(n)/size(grouped_train_sample{n},1);
    percent_error_test(n) = count_test_correct(n)/size(grouped_test_sample{n},1);
    
end

%%  Table
Num_labels_train = zeros(1,10);
Num_labels_test = zeros(1,10);
for n =  1:10
   Num_labels_test(n) = size(grouped_test_sample{n}, 1);
   Num_labels_train(n) = size(grouped_train_sample{n}, 1);
end
accuracy_table_Naive = table([0;1;2;3;4;5;6;7;8;9],Num_labels_test', count_test_correct', percent_error_test',Num_labels_train', count_train_correct',percent_error_train');
accuracy_table_Naive.Properties.VariableNames = {'label','N_labels_test','N_correct_test','test accuracy','N_labels_train','N_correct_train','train accuracy'}

overall_accuracy_Naive = table({'test','train'}',[10000;60000],[sum(count_test_correct);sum(count_train_correct)],[sum(count_test_correct)/10000 sum(count_train_correct)/60000]');
overall_accuracy_Naive.Properties.VariableNames = {'Sample Set', 'total Labels','correct classification','Overall Accuracy'}

%% Functions

% Vector computation instead of loop-based
function posterior_result = posterior2(mu_n,sigma_n,sample_vector)
    temp = 1./(sigma_n.*sqrt(2*pi)) .* exp(-.5.*((sample_vector - mu_n)./sigma_n).^2);
    posterior_result = sum(log(temp),'omitnan');
end

% calculate the likelihood of p(c|x) of all labels, then find the max
function Result = Naive_Bayes_Classification(mu_n_set,sigma_n_set,sample_vector,Prior)
    N_classes = size(mu_n_set,2);
    likelihood_array = zeros(1,10);
    for n = 1:N_classes
        likelihood_array(n) = posterior2(mu_n_set{n},sigma_n_set{n},sample_vector) + log(Prior(n));
    end 
    [max_likelihood, index] = max(likelihood_array);
    Result = index - 1;
end
