%% ENEE436 Foundation of Machine Learning Project1 
% Yuchen Zhou 
% University of Maryland, College Park
% Fall 2020, 10/31/2020


%%
load('data/project1_sample_info.mat');
% Using ML estimation to compute mu and sigma of each pixel of each label
mu_n = cell(1,10);
sigma_n = cell(1,10);
temp = cell(1,10);
for n = 1:10
    mu_n{n} = mean(grouped_train_sample{n});
    temp{n} = sum((grouped_train_sample{n} - repmat(mu_n{n},size(grouped_train_sample{n},1),1)).^2);
    sigma_n{n} = sqrt(temp{n}./(size(grouped_train_sample{n},1)));
end
%Prior Probabiltiy
Prior_P = zeros(1,10);
for n = 1:10
    Prior_P(n) = size(grouped_train_sample{n},1)/num_train_samples;
end

index_sum  = zeros(1,784);
for n = 1:10
     index_sum = index_sum + (sigma_n{n} ~= 0);
end
%% select the pixels that doesn't have variance of 0
for n = 1: 10
    mu_n{n} = mu_n{n}(:,index_sum == 10);
    sigma_n{n} = sigma_n{n}(:,index_sum == 10);
    grouped_train_sample{n} = grouped_train_sample{n}(:,index_sum == 10);
    grouped_test_sample{n} = grouped_test_sample{n}(:,index_sum == 10);
end


%% Data
data_0{1} = grouped_train_sample{1};
data_0{2} = grouped_test_sample{1};
data_1{1} = grouped_train_sample{2};
data_1{2} = grouped_test_sample{2};
data_7{1} = grouped_train_sample{8};
data_7{2} = grouped_test_sample{8};
data_8{1} = grouped_train_sample{9};
data_8{2} = grouped_test_sample{9};
data_9{1} = grouped_train_sample{10};
data_9{2} = grouped_test_sample{10};

%% 0 & 9
data_0_9_train = [data_0{1}; data_9{1}]';
data_0_9_test = [data_0{2}; data_9{2}]';
label_0_9_train = [0 * ones(5923,1); 1*ones(5949,1)];
label_0_9_test = [0 * ones(980,1); 1*ones(1009,1)];
w = fisher_LDA(data_0_9_train', label_0_9_train);
reduced_samples = w' * data_0_9_train;
reduced_samples_test = w'*data_0_9_test;
percent_match_train_09 = Fisher_LDA_classifier(label_0_9_train',reduced_samples,0,9,5923,5949,Prior_P);
percent_match_test_09 = Fisher_LDA_classifier(label_0_9_test',reduced_samples_test,0,9,980,1009,Prior_P);

%% 0 & 8  
data_0_8_train = [data_0{1}; data_8{1}]';
data_0_8_test = [data_0{2}; data_8{2}]';
label_0_8_train = [0 * ones(size(data_0{1},1),1); 1*ones(size(data_8{1},1),1)];
label_0_8_test = [0 * ones(size(data_0{2},1),1); 1*ones(size(data_8{2},1),1)];
w = fisher_LDA(data_0_8_train', label_0_8_train);
reduced_samples = w' * data_0_8_train;
reduced_samples_test = w'*data_0_8_test;
percent_match_train_08 = Fisher_LDA_classifier(label_0_8_train',reduced_samples,0,8,size(data_0{1},1),size(data_8{1},1),Prior_P);
percent_match_test_08 = Fisher_LDA_classifier(label_0_8_test',reduced_samples_test,0,8,size(data_0{2},1),size(data_8{2},1),Prior_P);
%% 1 & 7

data_1_7_train = [data_1{1}; data_7{1}]';
data_1_7_test = [data_1{2}; data_7{2}]';
label_1_7_train = [0 * ones(size(data_1{1},1),1); 1*ones(size(data_7{1},1),1)];
label_1_7_test = [0 * ones(size(data_1{2},1),1); 1*ones(size(data_7{2},1),1)];
w = fisher_LDA(data_1_7_train', label_1_7_train);
reduced_samples = w' * data_1_7_train;
reduced_samples_test = w'*data_1_7_test;
percent_match_train_17 = Fisher_LDA_classifier(label_1_7_train',reduced_samples,1,7,size(data_1{1},1),size(data_7{1},1),Prior_P);
percent_match_test_17 = Fisher_LDA_classifier(label_1_7_test',reduced_samples_test,1,7,size(data_1{2},1),size(data_7{2},1),Prior_P);


%% Error Rate Table
table_ = table(['0 and 9';'0 and 8'; '1 and 7'],1 - [percent_match_train_09;percent_match_train_08;percent_match_train_17],1 - [percent_match_test_09;percent_match_test_08;percent_match_test_17]);
table_.Properties.VariableNames = {'Case','Training Error Rate', 'Testing Error Rate'};


%% Functions
function percent_match = Fisher_LDA_classifier(original_label,reduced_samples, label_1, label_2, Num_label_1, Num_label_2,Prior_P)
    reduced_label_1 = reduced_samples(1:Num_label_1);
    reduced_label_2 = reduced_samples(Num_label_1+1: Num_label_1+Num_label_2);
    [mu1,sigmasq1]=max_likelihood_estimation1D(reduced_label_1);
    [mu2,sigmasq2]=max_likelihood_estimation1D(reduced_label_2);
    syms x
    equation = log(1/(sqrt(2*pi*sigmasq1))*exp(-(x-mu1).^2/(2*sigmasq1))) + log(Prior_P(label_1+1)) == log(1/(sqrt(2*pi*sigmasq2))*exp(-(x-mu2).^2/(2*sigmasq2))) +  log(Prior_P(label_2+1));
    decision_b = vpasolve(equation,x);
    dec =double(decision_b(1));
 
    
    hold off
    scatter(reduced_label_1, zeros(1,Num_label_1), 'LineWidth',1);
    hold on
    scatter(reduced_label_2, 100*ones(1, Num_label_2),  'LineWidth',2);
    x_mu1 = linspace((mu1-4*sqrt(sigmasq1)),(mu1+4*sqrt(sigmasq1)),1000);
    x_mu2 = linspace((mu2-4*sqrt(sigmasq2)),(mu2+4*sqrt(sigmasq2)),1000);
    y_mu1 = 1/(sqrt(2*pi*sigmasq1))*exp(-(x_mu1-mu1).^2/(2*sigmasq1));
    y_mu2 = 1/(sqrt(2*pi*sigmasq2))*exp(-(x_mu2-mu2).^2/(2*sigmasq2));
    plot(x_mu1, y_mu1, 'g-');
    plot(x_mu2, y_mu2, 'c--');
    xline(dec);
    
    prediction = (reduced_samples >= dec );
    percent_match = sum(prediction == original_label)/(Num_label_1 + Num_label_2);
    
end

function w=fisher_LDA(x,omega)
    %separting the samples of the two classes
    x_1=x(find(omega==1),:);
    x_2=x(find(omega==0),:);
    % calculating the necessary matrices
    mu1=mean(x_1);
    mu2=mean(x_2);
    S1=(x_1-repmat(mu1,size(x_1,1),1))'*(x_1-repmat(mu1,size(x_1,1),1));
    S2=(x_2-repmat(mu2,size(x_2,1),1))'*(x_2-repmat(mu2,size(x_2,1),1));
    Sw=S1+S2;
    w=inv(Sw)*(mu1-mu2)';
end


function [mu,sigma_sq]=max_likelihood_estimation1D(x)
    mu=mean(x);
    sigma_sq=var(x);
end
