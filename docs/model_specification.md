# Frozen model contract (candidate v0)

This document records the mathematical interface implemented in
`src/pd_project`. Numerical bounds and execution settings remain candidates
until Gate 2 is signed.

## Trial coding

Keep `raw_action` unchanged. Define a choice only when the source action is 1
or 2:

\[
y_i = \mathbf{1}(\text{raw\_action}_i=2),
\]

so `y=1` denotes the uncertain option. The choice and RT masks are separate:

\[
I_{y,i}=\mathbf{1}(\text{raw\_action}_i\in\{1,2\}),
\]

\[
I_{T,i}=I_{y,i}\,\mathbf{1}(T_i\text{ is finite})\,\mathbf{1}(T_i>0).
\]

## Valuation

The fixed amount scale is calculated from run A and expected to equal 10:

\[
s_0=\operatorname{median}_{i\in A}|r_{\mathrm{cert},i}|.
\]

With `p` expressed on the unit interval and odds against
\(O_i=(1-p_i)/p_i\):

\[
V_{\mathrm{cert},i}=r_{\mathrm{cert},i}/s_0,
\qquad
V_{\mathrm{uncert},i}
=\frac{r_{\mathrm{uncert},i}/s_0}{1+k_{j,c}O_i},
\]

\[
\Delta V_i=V_{\mathrm{uncert},i}-V_{\mathrm{cert},i}.
\]

Reward and loss have distinct positive `k` parameters. Loss amounts retain
their negative sign, so positive \(\Delta V\) always means the uncertain option
has the higher subjective value.

## Choice rule

\[
P(y_i=1)=\operatorname{logit}^{-1}(\beta_j\Delta V_i),
\qquad \beta_j>0.
\]

The choice-only baseline uses parameters
\((\log k_{j,R},\log k_{j,L},\log\beta_j)\) and is fit once. It is identical to
the full models except that the RT likelihood term is absent.

## RT observation model

All models use:

\[
\log T_i\sim\mathcal N(\mu_{i,m},\sigma_j^2),
\]

\[
\mu_{i,m}=\alpha_j+\delta_{j,L}\mathbf{1}(c_i=L)-b_{j,m}g_m(V_i),
\qquad b_{j,m}>0.
\]

The competing mappings are:

\[
g_1(V_i)=|\Delta V_i|,
\qquad
g_2(V_i)=(\Delta V_i)^2,
\qquad
g_3(V_i)=\frac{|V_{\mathrm{uncert},i}|+|V_{\mathrm{cert},i}|}{2}.
\]

The full parameter vector is:

\[
(\log k_R,\log k_L,\log\beta,\alpha,\delta_L,\log b,\log\sigma).
\]

## Joint likelihood

The working model assumes conditional independence of choice and RT given the
latent values and parameters:

\[
y_i\perp T_i\mid\Delta V_i,\theta_{j,m}.
\]

No arbitrary weighting is applied between modalities. The seconds-scale
log-normal score includes the Jacobian:

\[
\log f(T_i)=-\log T_i-\log\sigma-	frac12\log(2\pi)
-\frac{(\log T_i-\mu_i)^2}{2\sigma^2}.
\]

## Legal comparisons

- Compare M1-M3 on held-out run-B RT mean log predictive density.
- Compare each RT-informed fit with the single choice-only baseline on
  held-out choice mean log predictive density.
- Compare common-parameter recovery using paired errors.
- Compare reliability using paired differences in ICC(A,1), supported by
  Spearman correlations and Bland-Altman diagnostics.
- Never compare the full joint total likelihood directly with the choice-only
  total likelihood.

