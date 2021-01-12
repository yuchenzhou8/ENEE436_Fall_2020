%% 
%% KNN Plot
% Match_Percentatge vs. Number of Neighbors
figure (1)
hold off
plot(K,1-percent_match,'--g');
hold on
plot(K,1-percent_match_train,'-m');
scatter(K,1-percent_match, 'or');
scatter(K,1-percent_match_train,'*b');
xlabel('Number of Neighbors Considered');
ylabel('Error Rate');
title('Error Rate vs. K');
legend('Test Sample Match','Train Sample Match', 'Test Sample Match Data', 'Train Sample Match Data' );

%% KNN Table
result_table = table(K',(1-percent_match)', (1-percent_match_train)');
result_table.Properties.VariableNames ={'K','Test Match', 'Train Match'}