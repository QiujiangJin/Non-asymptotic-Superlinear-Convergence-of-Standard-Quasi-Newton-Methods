%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%

d = 30;

x_0 = 0.95*ones(d, 1);

x_opt = zeros(d, 1);

M = sqrtm(loss_hessian(x_opt));

disp("Initialization Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Gradient Descent %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_gd = [];
x = x_0;

for iter = 1:25
    x_new = x - 0.001*loss_gradient(x);
    x = x_new;
    y_gd = [y_gd, (norm(M*(x - x_opt))/norm(M*(x_0 - x_opt)))^(1/iter)];
end

disp("Gradient Descent Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Quasi Newton %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_qn = [];
x = x_0;
B = loss_hessian(x);

for iter = 1:25
    x_new = x - B\loss_gradient(x);
    I = eye(d);
    s = x_new - x;
    y = loss_gradient(x_new) - loss_gradient(x);
    B = B - B*s*s'*B/(s'*B*s) + y*y'/(s'*y);
    x = x_new;
    y_qn = [y_qn, (norm(M*(x - x_opt))/norm(M*(x_0 - x_opt)))^(1/iter)];
end

disp("Quasi-Newton Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Newton %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_n = [];
x = x_0;

for iter = 1:25
    B = loss_hessian(x);
    x_new = x - B\loss_gradient(x);
    x = x_new;
    y_n = [y_n, (norm(M*(x - x_opt))/norm(M*(x_0 - x_opt)))^(1/iter)];
    if y_n(iter) == 0
        y_n(iter) = 1e-6;
    end
end

disp("Newton Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = 1:25;
y = (1./sqrt(x));

semilogy(x, y_qn, '-.*r', 'LineWidth', 2);

hold on
semilogy(x, y_n, '-.*g', 'LineWidth', 2);
semilogy(x, y_gd, '-.*m', 'LineWidth', 2);
semilogy(x, y, '-.*b', 'LineWidth', 2);
l = legend({'Quasi-Newton', 'Newton', 'Gradient Descent', '$\frac{1}{\sqrt{k}}$'}, 'Location', 'southeast');
set(l, 'interpreter', 'latex')
xlabel('Number of iterations $k$','Interpreter','latex', 'fontsize', 18);
ylabel('$\left(\frac{\|\nabla^2{f(x_*)}^{\frac{1}{2}}(x_k - x_*)\|}{\|\nabla^2{f(x_*)}^{\frac{1}{2}}(x_0 - x_*)\|}\right)^{\frac{1}{k}}$', 'Interpreter', 'latex', 'fontsize', 18);
xlim([1 25]);
ylim([1e-5 1e0]);
hold off