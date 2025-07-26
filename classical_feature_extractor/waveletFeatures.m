function [cD1,cD2,cD3,cD4,cD5,cD6,cD7,cD8,cA8,D1,D2,D3,D4,D5,D6,D7,D8,A8] = waveletFeatures(s)
disp(size(s));
waveletFunction = 'db8';
[C,L] = wavedec(s,8,waveletFunction);
cD1 = detcoef(C,L,1); %detail coefficients lvl 1
cD2 = detcoef(C,L,2);%detail coefficients lvl 2
cD3 = detcoef(C,L,3);%detail coefficients lvl 3
cD4 = detcoef(C,L,4);%detail coefficients lvl 4
cD5 = detcoef(C,L,5); %GAMA%detail coefficients lvl 5
cD6 = detcoef(C,L,6); %BETA%detail coefficients lvl 6
cD7 = detcoef(C,L,7); %ALPHA%detail coefficients lvl 7
cD8 = detcoef(C,L,8); %THETA%detail coefficients lvl 8
cA8 = appcoef(C,L,waveletFunction,8); %DELTA %approximation coefficients lvl 8
D1 = wrcoef('d',C,L,waveletFunction,1); %Reconstruct of the detail coefficients at level 1.
D2 = wrcoef('d',C,L,waveletFunction,2);%Reconstruct of the detail coefficients at level 2.
D3 = wrcoef('d',C,L,waveletFunction,3);%Reconstruct of the detail coefficients at level 3.
D4 = wrcoef('d',C,L,waveletFunction,4);%Reconstruct of the detail coefficients at level 4.
D5 = wrcoef('d',C,L,waveletFunction,5); %Reconstruct of the detail coefficients at level 5.
D6 = wrcoef('d',C,L,waveletFunction,6); %Reconstruct of the detail coefficients at level 6.
D7 = wrcoef('d',C,L,waveletFunction,7); %%Reconstruct of the detail coefficients at level 7.
D8 = wrcoef('d',C,L,waveletFunction,8); %Reconstruct of the detail coefficients at level 8.
A8 = wrcoef('a',C,L,waveletFunction,8); %Reconstruct of approximation coefficients at level 8.
end
