clc
clear
close all

path1 = 'C:\Users\Maja\Desktop\po\zad 1\paper';
path2 = 'C:\Users\Maja\Desktop\po\zad 1\rock';
path3 = 'C:\Users\Maja\Desktop\po\zad 1\scissors';

x1 = dir('paper');
x2 = dir('rock');
x3 = dir('scissors');
papircnt = 0;
kamencnt = 0;
makazecnt = 0;
N = 712;
N_test = round(0.2*(N-3));
N_train = round(0.8*(N-3));
papir_feature = zeros(2,N);
kamen_feature = zeros(2,N);
makaze_feature = zeros(2,N);

for i = 3:N

    X1 = imread(fullfile(path1, x1(i).name));
    X2 = imread(fullfile(path2, x2(i).name));
    X3 = imread(fullfile(path3, x3(i).name));

    X1 = X1(10:end-10,25:end-20,:);
    X2 = X2(10:end-10,25:end-20,:);
    X3 = X3(10:end-10,25:end-20,:);

    j = i-2;
    papir_feature(1,j) = feature_extraction_1(X1);
    kamen_feature(1,j) = feature_extraction_1(X2);
    makaze_feature(1,j) = feature_extraction_1(X3);

    papir_feature(2,j) = feature_extraction_2(X1);
    kamen_feature(2,j) = feature_extraction_2(X2);
    makaze_feature(2,j) = feature_extraction_2(X3);

    papir_feature(3,j) = feature_extraction_3(X1);
    kamen_feature(3,j) = feature_extraction_3(X2);
    makaze_feature(3,j) = feature_extraction_3(X3);
end

figure
hold on
scatter(papir_feature(1,1:N_train),papir_feature(2,1:N_train),'rx')
scatter(kamen_feature(1,1:N_train),kamen_feature(2,1:N_train),'bx')
scatter(makaze_feature(1,1:N_train),makaze_feature(2,1:N_train),'gx')
hold off

X = [papir_feature(:,1:N_train), kamen_feature(:,1:N_train), makaze_feature(:,1:N_train)]';

Sigma = cov(X);
[F, L] = eig(Sigma);
Lambdas = [L(1,1), L(2,2), L(3,3)];
[Lambdas, indx] = sort(Lambdas, 'descend');
F = F(:,indx);
A = F(:,1:3); 

Y1 = papir_feature'*A;
Y2 = kamen_feature'*A;
Y3 = makaze_feature'*A;

figure();
hold all
scatter3(Y1(:,1), Y1(:,2), Y1(:,3), 'ro');
scatter3(Y2(:,1), Y2(:,2), Y2(:,3), 'bo');
scatter3(Y3(:,1), Y3(:,2), Y3(:,3), 'go');

legend('papir','kamen','makaze')


%% Konfuziona matrica

M_p = mean(Y1)'; S_p = cov(Y1);
M_k = mean(Y2)'; S_k = cov(Y2);
M_m = mean(Y3)'; S_m = cov(Y3);

len = round((N_test+1));
start = N_train+1;
Y=[Y1(start:start+len-1,:); Y2(start:start+len-1,:); Y3(start:start+len-1,:)];
Yp1 = ones(len,1);
Yp2 = ones(len,1)*2;
Yp3 = ones(len,1)*3;
Ytrue = [Yp1; Yp2; Yp3];
Yp=[];

for i=1:3*(N_test+1)
    x=Y(i,:)';
    fp=1/(2*pi*det(S_p)^1.5)*exp(-0.5*(x-M_p)'*inv(S_p)*(x-M_p));
    fk=1/(2*pi*det(S_k)^1.5)*exp(-0.5*(x-M_k)'*inv(S_k)*(x-M_k));
    fm=1/(2*pi*det(S_m)^1.5)*exp(-0.5*(x-M_m)'*inv(S_m)*(x-M_m));
    
    m=max([fp,fk,fm]);
    if m==fp
        predicted = 1;
    elseif m==fk
        predicted = 2;
    elseif m==fm
        predicted = 3;
    end
    Yp=[Yp; predicted];
end

m=confusionmat(Ytrue,Yp);
figure() 
confusionchart(m, categorical({'papir', 'kamen', 'makaze'}))
%%
Y1 = Y1';
Y2 = Y2';
K1 = zeros(2,N);
K2 = zeros(2,N);

K1(1,:) = Y1(1,1:N);
K1(2,:) = Y1(2,1:N);

K2(1,:) = Y2(1,1:N);
K2(2,:) = Y2(2,1:N);

%%

figure()
hist3(K1','Nbins', [30 30]);
figure();
hist3(K2','Nbins', [30 30]);


%%

K1 = K1';
K2 = K2';

M1_est = mean(K1)';
M2_est = mean(K2)';
S1_est = cov(K1);
S2_est = cov(K2);

s = 0:1e-3:1;
v0_opt_s = []; 
Neps_s = [];

for i= 1:length(s)
  
   V = ((s(i)*S1_est+(1-s(i))*S2_est)^(-1))*(M2_est-M1_est);

   Y1 = V'*K1';
   Y2 = V'*K2';
   Y = [Y1, Y2];
   Y = sort(Y);

   v0 = [];
   Neps = []; 
   for j = 1:(length(Y)-1)
      v0(j) = -(Y(j)+Y(j+1))/2; 
      Neps(j) = 0;
      for k = 1:N
          if Y1(k) > -v0(j)
              Neps(j) = Neps(j) + 1;
          end
          if Y2(k) < -v0(j)
              Neps(j) = Neps(j) + 1;
          end
      end
   end
   [Neps_s(i), index] = min(Neps);
   v0_opt_s(i) = v0(index);
end

[Neps_opt, index] = min(Neps_s);
v0_opt = v0_opt_s(index);
s_opt = s(index);


%% 
V = (s_opt*S1_est+(1-s_opt)*S2_est)^(-1)*(M2_est-M1_est);

K1 = K1'; 
K2 = K2';

figure
hold on
scatter(K1(1,:),K1(2,:),'ro')
scatter(K2(1,:),K2(2,:),'bo')

x1 = -1:0.01:1;
x2 = -(v0_opt+V(1)*x1)/V(2);
plot(x1,x2,'g', 'LineWidth', 2);
xlim([-1 0.3])
ylim([0 0.8])
legend('Papir', 'Kamen', 'Klasifikaciona linija')
hold off
