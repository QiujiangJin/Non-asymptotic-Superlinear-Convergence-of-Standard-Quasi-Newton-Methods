function res = loss_gradient(x)
    dim = size(x);
    d = dim(1);
    res = zeros(d, 1);
    res(1) = 400*x(1)^399 + 20000*x(1);
    for k = 2:d
        res(k) = 2*x(k);
    end
end