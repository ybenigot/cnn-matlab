
A=rand(32,32);
B=ones(2,2);
C=conv2(A,B,"valid");
d=size(C,1);
r=1+(0:d / 2)*2;
C1 = C(r,r);
size(C1)

C2=subsample(A,2);

% comupute what percentage of computation are close to each other, theoretically it should be 25*25
cnt=sum(sum( abs(C1-C2)<1e-6  ))

