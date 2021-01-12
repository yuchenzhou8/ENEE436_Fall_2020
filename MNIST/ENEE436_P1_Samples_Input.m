%% ENEE436 Foundation of Machine Learning Project1 
% Yuchen Zhou 
% University of Maryland, College Park
% Fall 2020, 10/31/2020

%% Load Training and Testing Samples
clear
train_data = csvread('data/mnist_train.csv');
test_data = csvread('data/mnist_test.csv');
%%
% each row of the mnist train/test file contains 1 + 28 * 28 = 785 digits
% number in the first column is the label in the image
% rest of the 784 digits represent the 28 * 28 pixels of each sample
[num_train_samples,label_pixels_train] = size(train_data);
[num_test_samples, label_pixels_test] = size(test_data);

% separate labels and image pixels by extract them from the data table
image_train = train_data(:,2:label_pixels_train);
image_test = test_data(:,2:label_pixels_test);
label_train = train_data(:,1);
label_test = test_data(:,1);

%% Group Samples by labels
% divide samples into 10 data groups based on their labels
grouped_test_sample = data_group(image_test,label_test,num_test_samples,10);
grouped_train_sample = data_group(image_train,label_train,num_train_samples,10);

%% Count Number of Images for Each Label
Num_labels_train = zeros(1,10);
Num_labels_test = zeros(1,10);
for n =  1:10
   Num_labels_test(n) = size(grouped_test_sample{n}, 1);
   Num_labels_train(n) = size(grouped_train_sample{n}, 1);
end


%% functions
function grouped_data = data_group(image_pixels,labels,num_samples,group)
    grouped_data = cell([1,group]);
    for n = 1:num_samples
        grouped_data{labels(n,1)+1} = [grouped_data{labels(n,1)+1}; image_pixels(n,:)];
    end
end