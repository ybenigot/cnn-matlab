function C=conv3(A,B)

% A a tensor of rank 3 whose two first dimensions are the image dimensions
% B a tensor of rank 3 whose two first dimensions are the image dimensions
% -> C a tensor whose two first dimensions are convolution images

	m=size(A,3);
	for i=1:m
		C(:,:,i)=conv2(A(:,:,i),B(:,:,i),'valid');
	end%for

end%function