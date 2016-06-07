function C=conv3full(A,B)

	m=size(A,3);
	for i=1:m
		C(:,:,i)=conv2(A(:,:,i),B(:,:,i),'full');
	end%for

end%function