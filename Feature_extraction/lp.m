%% 加载雷达数据
function data = lp(data,method)

if ~exist('method','var') || isempty(method)
  if isreal(data) && all(data(~isnan(data)) >= 0)
    % Data is probably magnitude or power (real and non-negative)
    method = 1;
  else
    % Data is either real w/ negative values or complex
    method = 2;
  end
end

if method == 1
  data = 10*log10(abs(data));
else
  data = 20*log10(abs(data));
end

return;

