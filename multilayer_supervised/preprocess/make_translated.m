function [data_out,labels_out]=make_translated(data_in,labels_in,trans_max,nb_trans,ratio)

% select 1/nb_trans of the data_in randomly and translate it
% data_out has ratio size relative to size as data_in

    indexes=randperm(size(data_in,2));
	indexes=indexes(1:ratio*size(data_in,2)/(1+nb_trans));
	data_selected=data_in(:,indexes);
	labels_selected=labels_in(indexes);

	n=sqrt(size(data_selected,1)); % image size

	data_translated=zeros(size(data_selected,1),size(data_selected,2)*nb_trans);
	labels_translated=zeros(size(data_selected,2)*nb_trans,1);

    fprintf('size of selected dataset  %d %d number of random translations %d\n',...
             size(data_selected,1),size(data_selected,2),nb_trans);
	k=1;
	for i=1:size(data_selected,2)
	    for j=1:nb_trans
	        translated = translate_image(reshape(data_selected(:,i),n,n),randperm(trans_max*2+1,2)-trans_max-1);
	        data_translated(:,k)=translated(:);
	        labels_translated(k)=labels_selected(i);
	        k=k+1;
	    end%for
	end%for

	data_out   = [data_translated,data_selected];
	labels_out = [labels_translated;labels_selected];


end%function