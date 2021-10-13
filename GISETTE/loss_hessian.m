function res = loss_hessian(X, w, mu)
    [m, d] = size(X);
    h = 1.0./(1.0 + exp(-X*w));
    I = eye(d);
    res = X'*(h.*(1 - h).*X)./m + mu*I;
end