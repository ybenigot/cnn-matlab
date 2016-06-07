function dump(text,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10)

% text : a text to print
% vars : a string listing some variable names
fprintf(text);
fprintf('\n');

n=1;
while exist(strcat('var',num2str(n)),'var') && n<=10
	eval(strcat('var',num2str(n)))
	n++;
endwhile

fprintf('\n');
%fflush(stdout);

end