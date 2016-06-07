function C=convolve3full(A,B)

% A a tensor of rank 3 whose two first dimensions are the image dimensions
% B a tensor of rank 3 whose two first dimensions are the image dimensions
% -> C a tensor whose two first dimensions are convolution images


% compute delta(i-r+1,j-r+1) theta (r,s)

%	A=cat(2,zeros(size(A,1),size(B,2)-1,size(A,3)),A);

	imax = size(A,1)+size(B,1)-1;
	jmax = size(A,2)+size(B,2)-1;
	R = size(B,1);
	S = size(B,2);

	C=zeros(imax,jmax,size(B,3));
	for i=1:imax
		for j=1:jmax
			i0=max(i-R+1,1);
			i1=min(i,size(A,1));
			j0=max(j-S+1,1);
			j1=min(j,size(A,2));
			%printf('values i0 %d i1 %d - j0 %d j1 %d\n',i0,i1,j0,j1);
			A2=A(i0:i1,j0:j1,:);
			A2=cat(1,zeros(size(B,1)-size(A2,1),size(A2,2),size(A2,3)),A2);			
			A2=cat(2,zeros(size(A2,1),size(B,2)-size(A2,2),size(A2,3)),A2);
			%whos		ac	
			C(i,j,:) = sum(sum(A2.*B,1),2);
		end%for
	end%for

end%function
