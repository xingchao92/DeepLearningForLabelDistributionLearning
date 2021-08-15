function Y = vl_klloss(x, x0,dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
%  modified 2015-09-07

% sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
% n = sz(4);
% index from 0
c(1,1,:,:) = x0' ;
if nargin <= 2 || isempty(dzdy) 
    t =  (c(1,1,:,:) +eps).* log((x)); % KL
   % t =  (c(1,1,:,:) .* log(x+eps) + x.* log(c+eps))./2; %symmetric KL
    Y =  -sum(t(:)) ;
else
    Y = -1./(x).*(dzdy*(c(1,1,:,:))); %
%     Y = -(1./(x+eps).*(dzdy*c(1,1,:,:))+ dzdy*log(c+eps))./2;
end