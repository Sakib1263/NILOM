function outp = jLWT(X,~)
lsc = liftingScheme('Wavelet','db2');
[outp,~] = lwt(X,'LiftingScheme',lsc,'Level',2);
end
