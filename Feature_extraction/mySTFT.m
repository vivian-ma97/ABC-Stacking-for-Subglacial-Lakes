%% ����ʱƵ�任
function [S,F,T]=mySTFT(tracedata,dt,varargin)
if isempty(varargin)
R=ceil((length(tracedata))/10);%��R�̶������ˣ��Ա�֤S������dt����ֱ���һ��
else
    R = varargin{1,1};
end

%��������������������������������������������ԭ������������������������������������
nfft=2^nextpow2(length(tracedata));
noverlap=floor(R-1);
[S,F,T,~]=spectrogram([zeros(1,ceil(noverlap/2)),tracedata,zeros(1,ceil(noverlap/2),1)],hanning(R), noverlap, nfft,1/dt);
S=abs(S);              %ȡSTFT�����ģ������ֵ����������ΪSTFT���ص��Ǹ��������ģֵ�����źŵķ�����Ϣ��




