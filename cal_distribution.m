feature = load('model/morph_50000/morph_feature.mat');
label_num = 60;
for i = 1:label_num
    tmp = mean(feature.ans.extractFeature(:,find(feature.ans.extractClass == i)),2);
    mean_feature(:,i) = tmp;
end
Dis = zeros(label_num,label_num);
for i =1:label_num
    for j =1:label_num
        Dis(i,j) = squaredxdist(mean_feature(:,i)',mean_feature(:,j)');
    end
end
Dis = Dis./max(max(Dis));
F = 1-Dis;
tmp = sum(F,2);
for i =1:label_num
    F(i,:) = F(i,:)./tmp(i);
end
save Distribution F;