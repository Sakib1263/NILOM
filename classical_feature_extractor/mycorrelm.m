function [newFeatureMatrix,newFeatureNames] = mycorrelm(feature_matrix, feature_list, threshold)
% Correlation
mycormat = corr(feature_matrix);
nameMask = ones(size(mycormat,1),1);
for i=1:length(mycormat)
    for j=i+1:length(mycormat)
        if mycormat(i,j) >= threshold
            if nameMask(j) == 1
                nameMask(j) = 0;
            end
        end
    end
end
% Remove highly correlated features
newFeatureMatrix = feature_matrix(:,nameMask == 1);
newFeatureNames = feature_list(nameMask == 1);
% Remove features with all zeroes
newFeatureMatrix = newFeatureMatrix(:,any(newFeatureMatrix,1));
newFeatureNames = newFeatureNames(1,any(newFeatureMatrix,1));
% Remove features with very low variance (e.g. all 1 or almost constant)
newFeatureMatrix_Temp = newFeatureMatrix;
newFeatureNames_Temp = newFeatureNames;
counter = 0;
for i=1:size(newFeatureMatrix,2)
    newFeatureMatrix_current = newFeatureMatrix(:,i);
    if var(newFeatureMatrix_current) > 0.0001
        counter = counter + 1;
        newFeatureMatrix_Temp(:,counter) = newFeatureMatrix(:,i);
        newFeatureNames_Temp(1,counter) = newFeatureNames(:,i);
    end
end
newFeatureMatrix = newFeatureMatrix_Temp(:,1:counter);
newFeatureNames = newFeatureNames_Temp(1,1:counter);
end
