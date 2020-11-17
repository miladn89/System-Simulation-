# System-Simulation-2
R Gibbs Sampler

```{r}

m = 100000; 
x = y = m; 
x[1] = y[1] = 0  # initial conditions
rho = 0.8; 
sgm = sqrt(1 - rho^2)                 # correlation and standard deviation

for (i in 2:m) {                                 # Gibbs Sampler Loop
  x[i] = rnorm(1, rho*y[i-1], sgm)
  y[i] = rnorm(1, rho*x[i], sgm)
}

c(mean(x), mean(y));
c(sd(x),sd(y));


a = .5
b = 1 - a
w = a*x + b*y

par(mfrow=c(2,2), pty="s") # 2 x 2 array of square plots
plot(x[1:50], y[1:50], type="l")
plot(x, y, pch=".", main = "x,y returns, green = portfolio > 0");
points(x[w>0], y[w>0], pch=".", col="green4")
plot(x[1:1000], type="l"); 
acf(y)

```

# Example 2: Bayesian Inference for Spam Filter

- Consider a spam filter and use a Bayesian Inference model to describe the process. 
- The goal is to estimate the prevalence $\pi$ of spam. 
- The model is represented:

Events:

- ${S=1}$: Email is in fact Spam. 
- ${S=0}$: Email is not Spam
- ${R=1}$: Email is marked as Spam.
- ${R=0}$: Email is marked as NOT Spam.
 
- We begin with a non-informative prior $Beta(\alpha=1,\beta=1)$. 
- In the previous class, we show that its posterior distribution with a binomial likelihood is still a Beta distribution with updated parameters. (Binomial and Beta are called conjugate pairs.)

New Data
- Total emails: $n=1,000$ emails.
- Emails marked as spam: $r = 200$.

Posterior distribution
- $\alpha_n = \alpha_0 + r$
- $\beta_n = \beta_0 + r$

What is the chance that an email will be rejected in future?
```{r}
alpha.0 = 1; beta.0 = 1      # Prior Information
n = 1000; r = 200            # Data

alpha.n = alpha.0 +r           # Posterior parameters
beta.n = beta.0 + n - r

alpha.n/(alpha.n+beta.n)     # Posterior density mean
```

What is the chance that a future email is spam?

- Let $\phi = P(S = 1)$. An incoming email is spam.
- Let $p = P\{S = 1 | R = 1 \}$. A rejected email is spam.
- Let $q = P\{S = 1 | R =0 \}$. Missed spam.
- Let $X$ be the number among the $r$ rejected emails that are indeed spam and $Y$ be the number of spam emails among the $n-r$ emails passed through. We can simulate these latent counts $X$ and $Y$ using Gibbs Sampling
- 1. $X|r,\phi \sim Binom(r, p)$
- 2. $Y | r, \phi \sim Bino(n-r, q)$
- 3. $\phi | X,Y \sim Beta(\alpha_0+X+Y, \beta_0+n-X-Y)$

```{r}
N = 100            # Number of iterations
Nb = 20; N1 = N+1  # Burn-in
eta = 0.99; theta = 0.97; 

psi = numeric(N) 
psi[1] = .5          # Initial value

                     # Gibbs Sampler Loop
for(i in 2:N) {
tau=psi[i-1]*eta+(1-psi[i-1])* (1-theta)        
X = rbinom(1, r, psi[i-1]* eta/tau)
Y = rbinom(1, n-r, psi[i-1]*(1-eta)/(1-tau))
psi[i] = rbeta(1, alpha.0+X+Y, beta.0+n-X-Y)
}

mean(psi[Nb:N])
```
```{r}
par(mfrow=c(2,2))
hist(psi)
plot(1:N,cumsum(psi)/(1:N),type="l",ylab= "psi", ylim=c(0.20,0.24))
plot(psi,type='p',pch='.',ylim=c(0.15,0.30))
acf(psi)
```

##Example 1: Issues with numerical integration. 
$x_1, \ldots, x_{10}$'s are sampled from a Cauthy distribution. Calculate the integration below:
\[
m(x) = \int^{\infty}_{-\infty} \prod^{10}_{i=1} \frac{1}{\pi} \frac{1}{1+(x_i - \theta)^2} d\theta
\]
for $\theta=350$.

```{r}
cac = rcauchy(10) + 350
lik = function(the){
  u = dcauchy(cac[1] - the)
  for (i in 2:10)
    u = u*dcauchy(cac[i] - the)
  return(u)}
integrate(lik, -Inf, Inf)
```
Is this numerical value correct?



## Example 2: Estimating the normal cdf $\Phi(x)$. 
The columns represent different values of x for $\Phi(x)$ and the rows represent various numbers of samples used to estimate the cdf.
```{r}
x = rnorm(10^8)
bound = qnorm(c(.5, .75, .8, .9, .95, .99, .999, .9999))
res = matrix(0, ncol = 8, nrow = 7)
for (i in 2:8)
  for (j in 1:8)
    res[i-1, j] = mean(x[1:10^i] < bound[j])
matrix(as.numeric(format(res, digi=4)), ncol = 8)
```

- It requires $10^8$ samples to obtain accurate estimates. 
- Importance Sampling can be use to reduce the coomputational costs.

```{r}
Nsim = 10^3
y=rexp(Nsim) + 4.5
weit = dnorm (y) / dexp(y-4.5)
plot(cumsum(weit)/1:Nsim, type = "l")
abline(a=pnorm(-4.5), b=0, col="red")
```

## Example 3: EM algorithm (Censored Data)
As we discussed in class, an analytical expression can be obtained for the iterations of $\theta_{(j)}$.

```{r}
y = c(1, 1.5, 0.5, 2, 2.5, 2)
m = length(y)
n=10
ybar = mean(y)
theta = rnorm(1, mean = ybar, sd = sd(y))
iter = 1
a = 3
while (iter < 50) {
  theta = c(theta, m * ybar/n + (n-m) * (theta[iter] + dnorm(a-theta[iter])/pnorm(a-theta[iter]))/n)
iter = iter +1
}
plot(theta)
```
