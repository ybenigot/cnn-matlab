function t=translate_image(image,trans)
% translates an image by a vector, replaces missing pixels by zero
% parameters
%	image 		an n x m matrix
% 	trans		a vector of dimension 2, each coordinate should be less than the image dimension

s=size(image);

if trans(1)>0
	image = [zeros(trans(1),s(2));image];%    prepad(image,trans(1)+size(image,1));
	range1=1:s(1);
elseif trans(1)<0
	image = [image;zeros(-trans(1),s(2))];%postpad(image,-trans(1)+size(image,1));
	range1=-trans(1)+1:size(image,1);
else
	range1=1:s(1);
end%if 

s=size(image);

if trans(2)>0
	image = [zeros(s(1),trans(2)),image];%prepad(image,trans(2)+size(image,2),0,2);
	range2=1:s(2);
elseif trans(2)<0
	image = [image,zeros(s(1),-trans(2))];%postpad(image,-trans(2)+size(image,2),0,2);
	range2=-trans(2)+1:size(image,2);
else	
	range2=1:s(2);
end%if 

t=image(range1,range2);

end%function