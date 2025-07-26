function [fig]= plotfeatureimportance(scores, ranking, names, method)

    %% Number of features to plot
    N = 10;
    
    %%
    figure;
    barh(diag(scores(ranking(1:N))),'stacked');
    title(['Feature Importance Estimates using ', method]);
    ylabel('Features');
    xlabel('Relative Importance');
    yt = 1:N;
    set(gca, 'YTick', yt,'YTickLabel',string(names(1:N)'));
    fig = gcf;
end

