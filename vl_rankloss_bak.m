function Y = vl_rankloss(x, x0, dzdy)
% Author: Xing Chao
% Email: xingchao@seu.edu.cn
% modified 2015-12-29
sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
%{
P_ = zeros(sz(3),sz(3),sz(4),'gpuArray');
for n=1:sz(4)
	for i=1:sz(3)
        for j=i+1:sz(3)
            if abs(i-x0(n))<=abs(j-x0(n))
                P_(i,j,n) = x(1,1,j,n) - x(1,1,i,n);
            else
                P_(i,j,n) = x(1,1,i,n) - x(1,1,j,n);
          	end 
        end
	end
end
%}
if nargin <= 2 || isempty(dzdy) 
    Y = 0;
    for n=1:sz(4)
        for i=1:sz(3)
            for j=i+1:sz(3)
                tmp = 0;
                if abs(i-x0(n))<abs(j-x0(n))
                    tmp = 1 + x(1,1,j,n) - x(1,1,i,n); 
                elseif abs(i-x0(n))==abs(j-x0(n))
                    tmp = abs(x(1,1,j,n) - x(1,1,i,n));     
                else
                    tmp = 1 + x(1,1,i,n) - x(1,1,j,n); 
                end 
                Y = Y + max(0,tmp);
            end
        end
        %{
        %basic rank entropy loss;
        i = x0(n);
        if i~=1
        	j = 1;
        else
            j = 10;
        end
        Y = Y + 0.5*(1-S_(n,i,j))*a*(x(1,1,i,n)-x(1,1,j,n)) + log( 1 + exp( -a*( x(1,1,i,n)-x(1,1,j,n))));
        %}
    end
    
else
    Y = zeros(sz,'gpuArray');
    
    for n=1:sz(4)
        %{
        basic rank entropy loss;
        i = x0(n);
        if i~=1
        	j = 1;
        else
           j = 10;
        end
        Y(1,1,i,n) = a*( 0.5*(1-S_(n,i,j)) - 1/(1+exp(-a*(x(1,1,i,n)-x(1,1,j,n)))));
        Y(1,1,j,n) = - a*( 0.5*(1-S_(n,i,j)) - 1/(1+exp(-a*(x(1,1,i,n)-x(1,1,j,n)))));
        %}
        
        for i=1:sz(3)
            for j=1:sz(3)
                if j==i continue; end;
                tmp = 0;
                tmp1 = 0;
                if abs(i-x0(n))<abs(j-x0(n))
                    tmp = 1 + x(1,1,j,n) - x(1,1,i,n);
                    tmp1 = -1;
                elseif abs(i-x0(n))==abs(j-x0(n))
                    tmp = abs(x(1,1,j,n) - x(1,1,i,n));
                    tmp1 = sign(x(1,1,i,n) - x(1,1,j,n));
                else
                    tmp = 1 + x(1,1,i,n) - x(1,1,j,n); 
                    tmp1 = 1;
                end 
                Y(1,1,i,n) = Y(1,1,i,n) + sign(max(0,tmp))*tmp1;
            end
        end
    end
    Y = Y .* dzdy;
end