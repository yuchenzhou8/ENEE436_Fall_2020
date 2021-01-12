%% ENEE436 Foundation of Machine Learning Project1 
% Yuchen Zhou 
% University of Maryland, College Park
% Fall 2020, 10/31/2020

%%
clear
load ('data/project1_sample_info.mat');
load ('data/color');
%%
sample_mu_train = mean(image_train); % compute the sample means of each pixel in the train set
sample_mu_test = mean(image_test); % compute the sample means of each pixels in the test set
%%  
PCA_coeff_all = pca(image_train);  % compute the eigenvectors/PCA coefficients for each PCA components in descending order 

%% N_component = 2
PCA_projection_2_train = cell(1,10);
PCA_projection_2_test = cell(1,10);
for n = 1:10
    PCA_projection_2_train{n} =  PCA_coeff_all(:,1:2)' *(grouped_train_sample{n} - repmat(sample_mu_train,Num_labels_train(n),1))';
    PCA_projection_2_test{n} =  PCA_coeff_all(:,1:2)' *(grouped_test_sample{n} - repmat(sample_mu_test,Num_labels_test(n),1))';
end
%%  N_component = 2 scatter
% scatter plot of projected samples
figure (1)
hold off 
for n = 1:10
    scatter(PCA_projection_2_train{n}(1,:),PCA_projection_2_train{n}(2,:),5,c{n},'filled');
    hold on
end
xlabel('PC1');
ylabel('PC2');
title('N component = 2, Train');
legend('0','1','2','3','4','5','6','7','8','9');

figure (2)
hold off 
for n = 1:10
    scatter(PCA_projection_2_test{n}(1,:),PCA_projection_2_test{n}(2,:),5,c{n},'filled');
    hold on
end
xlabel('PC1');
ylabel('PC2');
title('N component = 2, Test');
legend('0','1','2','3','4','5','6','7','8','9');

%% N_component = 3
PCA_projection_3_train = cell(1,10);
PCA_projection_3_test = cell(1,10);
for n = 1:10
    PCA_projection_3_train{n} =  PCA_coeff_all(:,1:3)' *(grouped_train_sample{n} - repmat(sample_mu_train,Num_labels_train(n),1))';
    PCA_projection_3_test{n} =  PCA_coeff_all(:,1:3)' *(grouped_test_sample{n} - repmat(sample_mu_test,Num_labels_test(n),1))';
end

%% N_component = 3 scatter
% scatter plot of projected samples N = 3
figure (3)
hold off 
for n = 1:10
    scatter3(PCA_projection_3_train{n}(1,:),PCA_projection_3_train{n}(2,:),PCA_projection_3_train{n}(3,:),5,c{n},'filled');
    hold on
end
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('N component = 3, Train');
legend('0','1','2','3','4','5','6','7','8','9');

figure (4)
hold off 
for n = 1:10
    scatter3(PCA_projection_3_test{n}(1,:),PCA_projection_3_test{n}(2,:),PCA_projection_3_test{n}(3,:),5,c{n},'filled');
    hold on
end
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('N component = 3, Test');
legend('0','1','2','3','4','5','6','7','8','9');

%% Naive Bayesian classification with PCA components = 5, 10, 20, 50,100

n = [5, 10, 20, 50, 100];
projected_sample_train = 0;
projected_sample_test = 0;
error_naive_train = zeros(1,5);
error_naive_test = zeros(1,5);
for N = 1:length(n)
    projected_sample_train = PCA_coeff_all(:,1:n(N))' *(image_train - repmat(sample_mu_train,num_train_samples,1))';
    projected_sample_test =  PCA_coeff_all(:,1:n(N))' *(image_test - repmat(sample_mu_test,num_test_samples,1))';
    Naive_Model = fitcnb(projected_sample_train',label_train); % Training Naive Model
    error_naive_train(N) = 1 - sum( predict(Naive_Model,projected_sample_train') ==  label_train)/60000;
    error_naive_test(N) = 1 -  sum( predict(Naive_Model,projected_sample_test') ==  label_test)/10000;
end

%% KNN classifcation with PCA components = 5, 10, 20, 50, 100
n = [5, 10, 20, 50, 100];
error_knn_train = zeros(1,5);
error_knn_test = zeros(1,5);
tic
for N = 1:length(n)
    projected_sample_train = PCA_coeff_all(:,1:n(N))' *(image_train - repmat(sample_mu_train,num_train_samples,1))';
    projected_sample_test =  PCA_coeff_all(:,1:n(N))' *(image_test - repmat(sample_mu_test,num_test_samples,1))';
    KNN_Model = fitcknn(projected_sample_train',label_train,'NumNeighbors',5,'distance','euclidean');  % Training KNN = 5 Model
    error_knn_train(N) = 1 - sum( predict(KNN_Model,projected_sample_train') ==  label_train)/60000;
    error_knn_test(N) = 1 - sum( predict(KNN_Model,projected_sample_test') ==  label_test)/10000;
end
toc

%%  Error Rate Plot
plot(n,error_knn_train,'-o','MarkerSize',5); 
hold on
plot(n,error_knn_test,'-o','MarkerSize',5);
plot(n,error_naive_train,'-o','MarkerSize',5);
plot(n,error_naive_test,'-o','MarkerSize',5);
hold off
xlabel('Number of PCA Components');
ylabel('Error Rate');
title('Error Rate of Different Classifiers vs. Number of PCA Compoents');
legend('KNN 5 Trainning Error rate','KNN 5 Testing Error rate','Naive Bayesian Trainning Error rate','Naive Bayesian Test Error rate');
