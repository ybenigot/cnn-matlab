function C=mean_upsample4(A,M,poolDim)

% A a tensor of rank 4 whose two first dimensions are the image dimensions
% M is anything, only there for compatibility for max pool upsampling, is not used
% poolDim the pool size for upsampling the images
% -> C a tensor whose two first dimensions are upsampled images

C=zeros(size(A,1)*poolDim,size(A,2)*poolDim,size(A,3),size(A,4));

for i=1:size(A,1)
	for j=1:size(A,2)
		for k=1:poolDim
			for l=1:poolDim
				C((i-1)*poolDim+k,(j-1)*poolDim+l,:,:) = (1/poolDim^2) * A(i,j,:,:);
			end%for
		end%for
	end%for
end%for

end%function	