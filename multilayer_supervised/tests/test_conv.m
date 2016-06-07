function test_conv()

	dim1=29;
	dim2=9;
	n = 1000;

	for a=1:10

		I=rand(dim1,dim1,n);
		F=rand(dim2,dim2,n);

		P1=convolve3(I,F);

		P2=zeros(dim1-dim2+1,dim1-dim2+1,n);
		for j=1:n
			P2(:,:,j)=conv2(I(:,:,j),rot90(F(:,:,j),2),'valid');
		end%for

		max(max(max(abs(P1-P2)))) % should be as low as 1e-13 for all values

	end%for

end%function