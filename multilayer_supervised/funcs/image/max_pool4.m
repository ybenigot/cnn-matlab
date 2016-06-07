function [C M]=max_pool4(A,poolDim)
% max pooling vectorized on samples and on filters
% A a tensor of rank 4 whose two first dimensions are the image dimensions
% poolDim the pool size for max on the images
% -> C a tensor whose two first dimensions are convolution images
% -> M a tensor whose first coordinate is for defining an index vector in the original images, then as C

C=zeros(  size(A,1)/poolDim,size(A,2)/poolDim,size(A,3),size(A,4));
M=zeros(2,size(A,1)/poolDim,size(A,2)/poolDim,size(A,3),size(A,4)); 

% iterate on all subimages to be pooled
for i=1:size(A,1)/poolDim
	for j=1:size(A,2)/poolDim
		% compute ranges of indes of subimages in the original images
		range1=(i-1)*poolDim+1:i*poolDim;
		range2=(j-1)*poolDim+1:j*poolDim;
		% max on dimension 3 of original tensor, keep index of max in IX
		[X IX]= max(A(range1,range2,:,:));
		% max on dimension 4 of original tensor, keep index in M(2,i,j:,:,:)
		[C(i,j,:,:) M(2,i,j,:,:)]= max(X);
		% the relevant value of IX is indexed by IY
		% iterate on dimensions 3 and 4 to synchronize these dimensions on M and IX
		for k=1:size(A,3)
			for l=1:size(A,4)
				M(1,i,j,k,l)=IX(1,M(2,1,1,k,l),k,l);
			end%for
		end%for
		% translate the index relative to the subimage coordinates to the original tensor coordinates
		M(1,i,j,:,:) = M(1,i,j,:,:) + (i-1)*poolDim;
		M(2,i,j,:,:) = M(2,i,j,:,:) + (j-1)*poolDim;
	end%for
end%for

end%function