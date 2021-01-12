/* 

    ENEE436 Foundation of Machine Learn 
    Project 1
    Yuchen Zhou
    README

*/


/
    Including:
        folders:
        -- data
            color.mat //  color cell array
            mnist_test.csv
            mnist_train.csv
            project1_knn_data.m
            project1_kNN_prediction.m
            project1_LDA.m
            project1_PCA.m
            project1_sample_info_mat
                
            mnist original binary files

            For convinence, I have saved the MATLAB workspace for
            KNN, LDA, and PCA. You may load them to skip the process time,
            espcially the long process time of KNN 

            Naive took shorter time to process, so I didn't save it
        
        -- old_code
            some old code I used for development
        -- report_images
            images I used in the report


        code:

        -- ENEE436_P1_Samples_Input.m
           (workspace is saved in project1_sample_info_mat）
           click run to run

        -- Naive_Classifcation_Prior.m 
           click run to run

        -- K_nearest_neighbor.m 
           (workspace is saved in project1_knn_data.m，
            project1_kNN_prediction.m）
           click run to run (not recommended, took too long)
           load('data/project1_kNN_prediction.m') (preferred)

        -- K_nearest_neighbor_output_plot.m (only plot the error rate)


        -- LDA.m 
           (workspace is saved in project1_LDA.m）
           click run to run

        -- PCA.m
           (workspace is saved in project1_PCA.m）
           click run to run (not recommended, knn takes some time)
           load('data/project1_PCA.m') (preferred)
           go to the section that plots, run section only



        
*/