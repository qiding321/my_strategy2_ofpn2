%% description
% y_t = a y_(t-1) + b + a_t
% a_t = sigma_t * epsilon_t
% sigma_t^2 = c + d * a_(t-1)^2 + e * sigma_(t-1)^2
a = .5;
b = .3;
c = .2;
d = .4;
e = .25;

%% data generating process
data_len = 10000;
epsilon = randn(data_len, 1);

y = nan(data_len, 1);
err = nan(data_len, 1);
sigma = nan(data_len, 1);

y(1) = 0;
err(1) = 0;
sigma(1) = 0;

for idx = 2 : data_len
    sigma(idx) = sqrt(c + d*err(idx-1)^2 + e * sigma(idx-1)^2);
    err(idx) = sigma(idx) * epsilon(idx);
    y(idx) = a*y(idx-1) + b + err(idx);
end

subplot(3, 1, 1)
plot(y);
subplot(3, 1, 2)
plot(err)
subplot(3, 1, 3)
plot(sigma)

%% garch estimation
mdl = arima(0, 0, 0);
mdl.Variance = garch(1, 1);
% mdl = arima('Variance', garch(1, 1));
estmdl = estimate(mdl, y(2:end), 'X', y(1:end-1));
% estmdl = estimate(mdl, y);


%% predict
[E, V, logL] = infer(estmdl, y(2:end), 'X', y(1:end-1));
[E2, V2, logL2] = infer(estmdl, ones(size(y, 1)-1, 1), 'X', y(1:end-1));

% aa = estmdl.AR{1};
aa = estmdl.Beta;
bb = estmdl.Constant;
cc = estmdl.Variance.Constant;
dd = estmdl.Variance.ARCH{1};
ee = estmdl.Variance.GARCH{1};

yy = nan(data_len, 1);
err2 = nan(data_len, 1);
sigma2 = nan(data_len, 1);
sigma_est = nan(data_len, 1);

yy(1) = 0;
err2(1) = 0;
sigma2(1) = 0;
sigma_est(1) = 0;

for idx = 2 : data_len
    err2(idx-1) = yy(idx-1)-y(idx-1);
    sigma2(idx) = sqrt(cc + dd*err2(idx-1)^2 + ee * sigma2(idx-1)^2);
    
    sigma_est(idx) = sqrt(cc + dd*err2(idx-1)^2 + ee * sigma2(idx-1)^2);
    yy(idx) = aa*y(idx-1) + bb;
end
scatter(y(2:end)-E, yy(2:end));
scatter(sigma2(2:end), sqrt(V), '.')
scatter(sqrt(V), abs(err(1:end-1)), '.')
