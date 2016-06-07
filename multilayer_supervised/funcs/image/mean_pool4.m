function [C M]=mean_pool4(A,poolDim)

% A a tensor of rank 4 whose two first dimensions are the image dimensions
% poolDim the pool size for averaging the images
% -> C a tensor whose two first dimensions are convolution images

C=zeros(size(A,1)/poolDim,size(A,2)/poolDim,size(A,3),size(A,4));
scale=1/poolDim^2;
for i=1:size(A,1)/poolDim
	for j=1:size(A,2)/poolDim
		range1=(i-1)*poolDim+1:i*poolDim;
		range2=(j-1)*poolDim+1:j*poolDim;
		C(i,j,:,:) = scale * sum(sum(A(range1,range2,:,:),1),2); 
	end%for
end%for
M=[]; %not relevant for mean

end%function