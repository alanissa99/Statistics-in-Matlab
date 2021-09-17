%% Problem 1: Simulering av konfidensintervall
close all, clear all, clc
% Parametrar:
n = 25; %Antal matningar
mu = 2; %Vantevardet
sigma = 1; %Standardavvikelsen
alpha = 0.05;
%Simulerar n observationer for varje intervall
x = normrnd(mu, sigma,n,100); %n x 100 matris med varden
%Skattar mu med medelvardet
xbar = mean(x); %vektor med 100 medelvarden.
%Beraknar de undre och ovre granserna
undre = xbar - norminv(1-alpha/2)*sigma/sqrt(n);
ovre = xbar + norminv(1-alpha/2)*sigma/sqrt(n);
%Ritar upp alla intervall
figure(1)
hold on
for k=1:100
if ovre(k) < mu % Rodmarkerar intervall som missar mu
plot([undre(k) ovre(k)],[k k],'r')
elseif undre(k) > mu
plot([undre(k) ovre(k)],[k k],'r')
 else
 plot([undre(k) ovre(k)],[k k],'b')
 end
 end
%b1 och b2 ar bara till for att figuren ska se snygg ut.
b1 = min(xbar - norminv(1 - alpha/2)*sigma/sqrt(n));
b2 = max(xbar + norminv(1 - alpha/2)*sigma/sqrt(n));
axis([b1 b2 0 101]) %Tar bort outnyttjat utrymme i figuren
%Ritar ut det sanna vardet
plot([mu mu],[0 101],'g')
hold off

%% Problem 2: Maximum likelihood/Minsta kvadrat
close all, clear all, clc
M = 1e5;
b = 4;
x = raylrnd(b, M, 1);
hist_density(x, 40)
hold on
my_est_ml =sqrt(sum(x.^2)/(2*M)) % Skriv in din ML-skattning har
my_est_mk =sum(x)*sqrt(2/pi)/(M) % Skriv in din MK-skattning har
plot(my_est_ml, 0, 'r*')
plot(my_est_mk, 0, 'g*')
plot(b, 0, 'ro')
hold on

plot(0:0.1:16, raylpdf(0:0.1:16, my_est_ml), 'r')
hold off
%Har ändrat från 0:0.1:6 till 0:0.1:16 för en bättre graf

%% Problem 3: Konfidensintervall for Rayleighfordelning
close all, clear all, clc
load wave_data.mat
subplot(2,1,1), plot(y(1:end))
subplot(2,1,2), hist_density(y)

my_est= sum(y)*sqrt(2/pi)/(length(y)) %Använder MK-skattning

%Hittar konf.-int. m.h.a paragraf 12.3 i F.S
s=sqrt( 1/(length(y)-1) * sum((y-mean(y)).^2) );
D=s/sqrt(length(y))*sqrt(2/pi);
alpha=0.05;
lambda=norminv(1-alpha/2);

lower_bound=my_est-lambda*D
upper_bound=my_est+lambda*D

hold on % Gor sa att ploten halls kvar
plot(lower_bound, 0, 'g*')
plot(upper_bound, 0, 'g*')

plot(0:0.1:6, raylpdf(0:0.1:6, my_est), 'r')
hold off

%% Problem 4: Jämförelse av fördelningar hos olika populationer
close all, clear all, clc
load birth.dat
vikt=birth(:, 3);
malder=birth(:, 4);
mvikt=birth(:, 15);
mlangd=birth(:, 16);

figure(1)
subplot(2,2,1),hist_density(vikt)
title('Barnetsfödelsevikt')
subplot(2,2,2), hist_density(malder)
title('Moderns ålder')
subplot(2,2,3), hist_density(mvikt)
title('Moderns vikt')
subplot(2,2,4), hist_density(mlangd)
title('Moderns längd')


figure(2)
x = birth(birth(:, 20) < 3, 3);
y = birth(birth(:, 20) == 3, 3);
subplot(2,2,1), boxplot(x),
axis([0 2 500 5000])
subplot(2,2,2), boxplot(y),
axis([0 2 500 5000])

subplot(2,2,3:4), ksdensity(x),
hold on
[fy, ty] = ksdensity(y);
plot(ty, fy, 'r')
hold off



figure(3)
w = birth(birth(:, 23) == 0, 3);
z = birth(birth(:, 23) == 1, 3);
subplot(2,2,1), boxplot(w), ylabel('vikt [g]'), title('Normal moder')
axis([0 2 500 5000])
subplot(2,2,2), boxplot(z), ylabel('vikt [g]'), title('Lätt moder')
axis([0 2 500 5000])
subplot(2,2,3:4), ksdensity(w),
hold on
[fy, ty] = ksdensity(z);
plot(ty, fy, 'r')
legend('Normal moder','Lätt Moder')
hold off

%% Problem 5: Test av normalitet
close all, clear all, clc

load birth.dat

vikt=birth(:,3);
malder=birth(:,4);
mlangd=birth(:,16);
mvikt=birth(:,15);

subplot(2,2,1),normplot(vikt), xlabel('Barnets födelsevikt [g]')
subplot(2,2,2),normplot(malder), xlabel('Moderns ålder [år]')
subplot(2,2,3),normplot(mlangd), xlabel('Moderns längd [cm]')
subplot(2,2,4),normplot(mvikt), xlabel('Modersn vikt [kg]')

vikt_test= jbtest(vikt)
malder_test= jbtest(malder)
mlangd_test= jbtest(mlangd)
mvikt_test= jbtest(mvikt)


%% Problem 6: Enkel linjär regression
close all, clear all, clc

load moore.dat

X=[ones(length(moore(:,1)),1), moore(:,1)];
y1=(moore(:,2));
beta_hat=regress(log(y1),X);

figure(1)
plot(X(:,2),X*beta_hat)
hold on
plot(X(:,2),log(y1),'x')

figure(2)
res = log(y1)-X*beta_hat;
subplot(2,1,1), normplot(res)
subplot(2,1,2), hist(res)


[B,BINT,R,RINT,STATS] = regress(log(y1),X);
R2=STATS(1)


Prediktion=exp(beta_hat(1) + beta_hat(2)*2025)


%% Problem 7: Multipel linjär regression
close all, clear all, clc

load birth.dat
x1=birth(:,16); %Moderns längd
X1=[ones(length(x1),1), x1  ];
y1=birth(:,3); %Barnets vikt
beta_hat=regress(y1,X1); 

figure(1) %Figur för vår enkel linjär regressionsmodell
plot(x1,X1*beta_hat)
hold on
plot(x1,y1,'x')


%Multipel linjär regressionsmodell

x=birth(:,15);
w=birth(:,20);
z=birth(:,23);

w(w<=2)=0; %Röker inte
w(w==3)=1; %Röker

[B,BINT]=regress(y1,[X1(:,1), x, w, z]);
BINT

res1=y1- ( B(1).*X1(:,1) + B(2).*x + B(3).*w +B(4).*z );

figure(2)
normplot(res1)



