t=time;
poolDim=2;
filter=ones(poolDim,poolDim);						% this is average pooling
product=conv2(a,filter,'valid');
productDim=size(product,1);
range1=1+(0:productDim / poolDim)*poolDim;
time-t
product(range1,range1) ./ (poolDim * poolDim)
% 4 times faster than subsample on single 4x4 matrix
