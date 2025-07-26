function outp = jEntropyWavelet(X,~)
[Wen1,~] = wentropy(X,Entropy="Shannon",Level=3,Scaled=1);
[Wen2,~] = wentropy(X,Entropy="Renyi",Exponent=2);
[Wen3,~] = wentropy(X,Entropy="Tsallis",Exponent=2);
outp = cat(1,Wen1,Wen2,Wen3);
end
