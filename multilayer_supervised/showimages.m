[~, ~, pred] = supervised_nn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);

for i=randperm(size(data_test,2),min(10,size(data_test,2)))
  fprintf('sample # : %d\n',i);
  fprintf('prediction : %d actual value : %d\n',pred(1,i)-1,labels_test(i)-1);
  imshow(reshape(data_test(:,i),28,28),'InitialMagnification','fit');
  pause(4);
  close;
end%for

