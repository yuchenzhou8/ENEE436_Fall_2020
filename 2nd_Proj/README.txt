/* 

    ENEE436 Foundation of Machine Learn 
    Project 2
    Yuchen Zhou
    README

*/

/****** all scripts are written and tested on MATLAB R2020a  ******/
/
    including:
        -- task1 (SVM)
            -- data_parse.m 
                -- parse and save data from .asc files
                -- run: click run
            -- SVM_Linear.m
                -- SVM models using the linear kernel
                -- run: click run
            -- SVM_RBF.m
                -- SVM models using the Gaussian RBF kernel
                -- Using cross validation of k = 5 to 
                -- find the optimal kernel parameter sigma
                -- run: click run
            -- some data samples

        -- task2 (Neural Network)
            -- ANN_2_output.m (data in the report are drawn from this script)
                -- neural network with two output node
                -- plots accuracy vs. number of hidden layer neurons
                -- run: click run (takes some time to train the models)
            -- ANN.m
                -- neural network with one output node
                -- plots accuracy vs. number of hidden layer neurons
                -- run: click run (takes some time to train the models)
            -- some data samples 

        -- task3 (Unsupervised Clustering)
            -- Gaussian_mixture.m
                -- Gaussian_mixture algorithm is applied to a set of unclassified
                -- data (assuming # of clusters = 2)
                -- Calculate the mean and covariance of each set of data
                -- plot the unclassified data with mean and variance
                -- run: click run
            -- K_means.m
                -- K means clustering is applied to sets of unclassified data
                -- with # of clusters = 2 ,3,4
                -- plot the classified data using K_means
                -- run: click run
            -- Spectral.m
                -- Self-Implemented Spectral clustering is applied to 
                -- sets of unclassified data
                -- with # of clusters = 2,3,4 (the 
                -- plot the classified data using Spectral
                -- run: click run
/