function res = loss_function(x)
    dim = size(x);
    d = dim(1);
    res = x(1)^400 + 10000*x(1)^2;
    for k = 2:d
        res = res + x(k)^2;
    end
end