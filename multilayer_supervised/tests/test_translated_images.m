[~, ~, pred] = supervised_nn_cost( opt_params, ei, data_trans, [], true);
[~,pred] = max(pred);
acc_trans = mean(pred'==labels_trans);
fprintf('trans accuracy: %f\n', acc_trans);

for i=randperm(size(data_trans,2),min(10,size(data_trans,2)))
  fprintf('sample # : %d\n',i);
  fprintf('prediction : %d actual value : %d\n',pred(1,i)-1,labels_trans(i)-1);
  imshow(reshape(data_trans(:,i),28,28));
  pause(4);
  close;
end%for

