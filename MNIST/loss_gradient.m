function res = loss_gradient(X, y, w, mu)
    dim = size(y);
    m = dim(1);
    h = 1.0./(1.0 + exp(-X*w));
    res = sum((h - y).*X);
    res = res'./m + mu*w;
end