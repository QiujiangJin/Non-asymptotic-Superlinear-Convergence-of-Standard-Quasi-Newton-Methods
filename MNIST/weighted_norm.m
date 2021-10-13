%%%%%%%%%%%%%%%%%%%%%% Train Data Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%

images = loadMNISTImages('Data/train-images-idx3-ubyte');
labels = loadMNISTLabels('Data/train-labels-idx1-ubyte');
images = images';

data = [];

[a,~] = find(labels==8);
[b,~] = find(labels==0);
c=[a;b];
 
label1 = ones(length(a),1);
label2 = -1*ones(length(b),1);

data = [data images(c,:)];
label = [label1 ;label2];

U =[data,label];
shuffledArray = U(randperm(size(U,1)),:);

X_train = shuffledArray(:,1:784);
X_train = normalize_row(X_train);
Y_train = shuffledArray(:,785);
Y_train = (Y_train + 1)/2;

[N, d] = size(X_train);

clear images labels N1 m1 data label a b c label1 label2 U shuffledArray;

%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = 0.01;
L = mu + 1;
w_0 = 0.1*ones(d, 1);
w = w_0;

epsilon = 1e-16;
for iter = 1:20
    w = w - loss_hessian(X_train, w, mu)\loss_gradient(X_train, Y_train, w, mu);
    if norm(loss_gradient(X_train, Y_train, w, mu)) < epsilon
        break;
    end
end

w_opt = w;
minimizer = loss_function(X_train, Y_train, w_opt, mu);
M = sqrtm(loss_hessian(X_train, w_opt, mu));

disp("Initialization Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Gradient Descent %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_gd = [1];
w = w_0;

for iter = 1:20
    w = w - (1./sqrt(iter))*loss_gradient(X_train, Y_train, w, mu);
    y_gd = [y_gd, norm(M*(w - w_opt))/norm(M*(w_0 - w_opt))];
end

disp("Gradient Descent Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% BFGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_bfgs = [1];
w = w_0;
H = inv(loss_hessian(X_train, w_0, mu));

for iter = 1:20
    w_new = w - H*loss_gradient(X_train, Y_train, w, mu);
    I = eye(d);
    s = w_new - w;
    y = loss_gradient(X_train, Y_train, w_new, mu) - loss_gradient(X_train, Y_train, w, mu);
    t = 1.0/(s'*y);
    G = t*(H*y)*s';
    K = s*s';
    H = H - G' - G + (t^2*(y'*H*y) + t)*K;
    w = w_new;
    y_bfgs = [y_bfgs, norm(M*(w - w_opt))/norm(M*(w_0 - w_opt))];
end

disp("BFGS Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% DFP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_dfp = [1];
w = w_0;
H = inv(loss_hessian(X_train, w_0, mu));

for iter = 1:20
    w_new = w - H*loss_gradient(X_train, Y_train, w, mu);
    I = eye(d);
    s = w_new - w;
    y = loss_gradient(X_train, Y_train, w_new, mu) - loss_gradient(X_train, Y_train, w, mu);
    t = 1.0/(s'*y);
    G = H*y;
    K = s*s';
    H = H - G*G'/(y'*G) + t*K;
    w = w_new;
    y_dfp = [y_dfp, norm(M*(w - w_opt))/norm(M*(w_0 - w_opt))];
end

disp("DFP Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Newton %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_n = [1];
w = w_0;

for iter = 1:20
    w = w - loss_hessian(X_train, w, mu)\loss_gradient(X_train, Y_train, w, mu);
    y_n = [y_n, norm(M*(w - w_opt))/norm(M*(w_0 - w_opt))];
    if y_n(iter) < 1e-15
        y_n(iter) = 1e-16;
    end
end

disp("Newton Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = 1:20;
y = [1, (1./sqrt(x)).^(x)];
x = 0:20;

semilogy(x, y_bfgs, '-.*r', 'LineWidth', 3);

hold on
semilogy(x, y_dfp, '-.*b', 'LineWidth', 3);
semilogy(x, y_n, '-.*g', 'LineWidth', 3);
semilogy(x, y_gd, '-.*y', 'LineWidth', 3);
semilogy(x, y, '-.*m', 'LineWidth', 3);
l = legend({'BFGS', 'DFP', 'Newton', 'Gradient Descent', '$\left(\frac{1}{\sqrt{k}}\right)^{k}$'});
set(l, 'Interpreter', 'latex', 'fontsize', 20, 'Location', 'northeast')
xlabel('Number of iterations $k$','Interpreter','latex', 'fontsize', 20);
ylabel('$\frac{\|\nabla^2{f(x_*)}^{\frac{1}{2}}(x_k - x_*)\|}{\|\nabla^2{f(x_*)}^{\frac{1}{2}}(x_0 - x_*)\|}$', 'Interpreter', 'latex', 'fontsize', 20);
xlim([0 20]);
ylim([1e-15 1e0]);
ax = gca;
ax.FontSize = 15;
set(gcf,'position',[0,0,600,400])
hold off