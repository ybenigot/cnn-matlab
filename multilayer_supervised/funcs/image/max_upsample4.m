function U=max_upsample4(A,M,poolDim)

% A a 4-tensor whose two first dimensions are the image dimensions
% poolDim the pool size for upsampling the images
% M a 5-tensor, first dim dirrentiates index rank of coordinate, four dimensions in A, 
%    value is index of max in upsampled image
% -> U a tensor whose two first dimensions are upsampled images

U=zeros(size(A,1)*poolDim,size(A,2)*poolDim,size(A,3),size(A,4));

for i=1:size(A,1)
	for j=1:size(A,2)
		for k=1:poolDim
			for l=1:poolDim				
				U((i-1)*poolDim+k,(j-1)*poolDim+l,:,:) = squeeze(A(i,j,:,:)) .*...
					squeeze(M(1,i,j,:,:)==(i-1)*poolDim+k) .*... % multiply by 1 if corresponding coordinate
					squeeze(M(2,i,j,:,:)==(j-1)*poolDim+l) ;     % zero therwise
			end%for
		end%for
	end%for
end%for

end%function