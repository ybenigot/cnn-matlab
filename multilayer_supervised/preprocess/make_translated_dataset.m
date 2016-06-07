% make a translated test dataset
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist(1); 

trans_max=4;
nb_trans=4;
m=size(data_test,2); 

[data_train,labels_train] = make_translated(data_train,labels_train,trans_max,nb_trans,1+nb_trans);
[data_test,labels_test] = make_translated(data_test(:,1:m/2),labels_test(1:m/2),trans_max,nb_trans,2);
[data_valid,labels_valid] = make_translated(data_test(:,1+m/2:end),labels_test(1+m/2:end),trans_max,nb_trans,2);

n=sqrt(size(data_train,1)); % image size

for j=randperm(100,10)
	imshow(reshape(data_train(:,j),n,n));
	pause(2);
	close;
end%for

save 'dataset/mnist_translated2.mat'
