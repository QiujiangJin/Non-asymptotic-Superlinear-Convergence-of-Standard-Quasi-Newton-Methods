function B = normalize_row(A)
    [m, d] = size(A);
    B = zeros(m ,d);
    for i = 1:m
        B(i, :) = A(i, :)./norm(A(i, :));
    end
end