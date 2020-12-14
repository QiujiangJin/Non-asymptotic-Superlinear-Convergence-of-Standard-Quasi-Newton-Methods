function res = loss_hessian(x)
    dim = size(x);
    d = dim(1);
    res = zeros(d, d);
    res(1, 1) = 1560*x(1)^38 + 200;
    for k = 2:d
        res(k, k) = 2;
    end
end