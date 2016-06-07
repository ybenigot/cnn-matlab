function C=convolve4(A,B)

% A a tensor of rank 4 whose two first dimensions are the image dimensions
% B a tensor of rank 4 whose two first dimensions are the image dimensions
% -> C a tensor whose two first dimensions are convolution images

imax = size(A,1)-size(B,1)+1;
jmax = size(A,2)-size(B,2)+1;

C=zeros(imax,jmax,size(B,3),size(B,4));
for i=1:imax
	for j=1:jmax
		C(i,j,:,:) = sum(sum(A(i:i+size(B,1)-1,j:j+size(B,2)-1,:,:).*B,1),2);
	end%for
end%for



end%function