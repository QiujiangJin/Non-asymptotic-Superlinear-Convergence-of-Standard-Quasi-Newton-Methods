function res = loss_function(X, y, w, mu)
    dim = size(y);
    m = dim(1);
    h = 1.0./(1.0 + exp(-X*w));
    res = - y'*log(h) - (1 - y)'*log(1 - h);
    res = res./m + mu*(w'*w)./2;
end