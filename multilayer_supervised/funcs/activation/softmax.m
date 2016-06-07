function h=softmax(z)


	h_num = exp(z);							% size layer_size x m
	h_den = sum(h_num,1);
    h = h_num ./ repmat(h_den,size(h_num,1),1);

end%function
