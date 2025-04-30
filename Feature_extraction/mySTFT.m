%% 计算时频变换
function [S,F,T]=mySTFT(tracedata,dt,varargin)
if isempty(varargin)
R=ceil((length(tracedata))/10);%把R固定下来了，以保证S矩阵在dt方向分辨率一致
else
    R = varargin{1,1};
end

%——————————————————能量集中原理——————————————————
nfft=2^nextpow2(length(tracedata));
noverlap=floor(R-1);
[S,F,T,~]=spectrogram([zeros(1,ceil(noverlap/2)),tracedata,zeros(1,ceil(noverlap/2),1)],hanning(R), noverlap, nfft,1/dt);
S=abs(S);              %取STFT结果的模（绝对值），这是因为STFT返回的是复数结果，模值代表信号的幅度信息。




