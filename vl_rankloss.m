function Y = vl_rankloss(x, x0, dzdy)
% Author: Xing Chao
% Email: xingchao@seu.edu.cn
% modified 2015-12-29
sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;

if nargin <= 2 || isempty(dzdy) 
    
    Y = 0;
    for n=1:sz(4)
       tmp = 1:sz(3);
       tmp = abs(tmp - x0(n));
       [tmpX,tmpY] = meshgrid(tmp);
       tmpC = sign(tmpX - tmpY);  
       
       [XX,YY] = meshgrid(x(1,1,:,n));
       XY = XX - YY;
       res = tmpC.*XY;
       Y = Y + 0.5*sum(sum((max(0,1+res))));
    end 
    %{
    Y = 0;
    for n=1:sz(4)
        for i=1:sz(3)
            for j=i+1:sz(3)
                tmp = 0;
                if abs(i-x0(n))<abs(j-x0(n))
                    tmp = x(1,1,j,n) - x(1,1,i,n); 
                elseif abs(i-x0(n))==abs(j-x0(n))
                    tmp = 0;     
                else
                    tmp = x(1,1,i,n) - x(1,1,j,n); 
                end 
                Y = Y + max(0,tmp);
            end
        end
    end
    %}
else
    
    Y = zeros(sz,'gpuArray');
    for n=1:sz(4)
       tmp = 1:sz(3);
       tmp = abs(tmp - x0(n));
       [tmpX,tmpY] = meshgrid(tmp);
       tmpC = sign(tmpX - tmpY);  
       
       [XX,YY] = meshgrid(x(1,1,:,n));
       XY = XX - YY;
       res = tmpC.*XY;
       Y(1,1,:,n) = sum(sign(max(0,1+res)).*(tmpC),1);
    end   
    Y = Y .* dzdy;
    
    %{
    Y2 = zeros(sz,'gpuArray');
    
    for n=1:1   
        for i=1:sz(3)
            for j=1:sz(3)
                if j==i continue; end;
                tmp = 0;
                tmp1 = 0;
                if abs(i-x0(n))<abs(j-x0(n))
                    tmp = x(1,1,j,n) - x(1,1,i,n);
                    tmp1 = -1;
                elseif abs(i-x0(n))==abs(j-x0(n))
                    tmp = 0;
                    tmp1 = 0;
                else
                    tmp = x(1,1,i,n) - x(1,1,j,n); 
                    tmp1 = 1;
                end 
                Y2(1,1,i,n) = Y2(1,1,i,n) + sign(max(0,tmp))*tmp1;
            end
        end
    end
    Y2 = Y2 .* dzdy;
    %}
end