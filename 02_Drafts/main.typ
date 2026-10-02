#import "@preview/ssrn-scribe:0.10.1": paper

#show: paper.with(
  meta: (
    title: [How costly are Scalable Markups?],
    authors: (
      (
        name: "Enrico Truzzi",
        affiliation: "Universitat Pompeu Fabra",
        email: "enrico.truzzi@upf.edu",
      ),
    ),
    abstract: [#h(1.5em)Standard oligopoly models assign every firm the same returns to scale, such that firms differ in productivity but not in how easily they can grow. This paper studies how the sorting between capability and scalability shapes endogenous markups and their aggregate welfare cost. I build a nested-CES Cournot economy in the spirit of #cite(<edmond2023costly>, form: "prose"), where firm-level returns to scale are disciplined externally by gross-output production-function estimates for Compustat firms, and where the alignment between scalability and capability is calibrated to the observed correlation between returns to scale and log sales ($0.596$). In the calibrated economy, moving from the market to the efficient allocation yields a consumption-equivalent gain of $31.9%$, most of which comes from the aggregate scale of production rather than from the reallocation of a given stock of inputs. Holding both marginal distributions fixed, randomly reassigning scalability across firms lowers this cost to $17.4%$, with sorting accounting for $42%$ of the total loss and for $91%$ of the loss due to markup dispersion. Additionally, I find that the welfare cost rises far faster with the correlation over positive than over negative values. Therefore, the cost of market power depends on which technologies carry the markup wedge, since a given wedge is more distortionary when it sits on firms that can expand output at little additional cost.],
    date: datetime.today().display("[month repr:long] [day padding:none], [year]"),
    keywords: [Markups, Returns to scale, Scalability, Sorting, Market power, Misallocation, Oligopoly],
  ),
  theme: (
    font: "New Computer Modern",
    heading-font: "New Computer Modern",
  ),
  layout: (
    maketitle: true,
    // The template leaves the abstract box (86% wide) left-aligned, which
    // looks off-centre; full text width keeps it flush with the margins.
    cover-text-width: 100%,
    cover-spacing: 28pt,
    frontmatter-gap: 12pt,
    density: "balanced",
    // No extra gap between paragraphs (same as the line leading); paragraphs
    // are separated by the first-line indent only.
    body-paragraph-spacing: 0.62em,
    // LaTeX-style indent: the first paragraph after a heading (or after an
    // equation, figure, or spacing) is not indented, later ones are.
    body-first-line-indent: (amount: 1.5em, all: false),
  ),
)

// Document-level rules carried over from model.typ.
#set math.vec(gap: 1em)
#set figure.caption(position: top)
// Keep the previous 1.2em breathing room around display equations and
// figures, which otherwise inherit the (now tight) paragraph spacing.
#show math.equation.where(block: true): set block(spacing: 1.2em)
#show figure: set block(spacing: 1.2em)

// Equation references render as the bare number, e.g. "Equation (3)" is written
// in the text as "Equation @eq:...".
#show ref: it => {
  let eq = math.equation
  let el = it.element
  if el != none and el.func() == eq {
    link(el.location(), numbering(
      el.numbering,
      ..counter(eq).at(el.location())
    ))
  } else {
    it
  }
}

*TODO:*

+ Clean Introduction and add literature relationship
+ Write Conclusion
+ Add abstract

*Research question:* _How does capability-scalability sorting shape endogenous
          markups and their aggregate welfare cost ?_
#v(0.5cm)
= Proto-Introduction
#v(0.2cm)

Classic oligopoly models typically assign every firm the same returns-to-scale parameters, so firms differ in productivity but not in how easily they can scale. With heterogenous RTS, capability and scalability jointly determine equilibrium size, and, under oligopoly, size determines markups. I argue that the sorting between productivity (here refered to as capability) and scalability , and not only their absolute value, plays a major role in shaping the aggregate welfare costs of market power. That is, the

The quantitative environment is an EMX-style sector oligopoly model with heterogeneous market exposure and finite-firm competition inside product markets. One simulated unit is one sector, and one sector is one nested-CES Cournot market. Markups are equilibrium outcomes,where more concentrated sectors generate larger firm market shares, and larger market shares relax perceived demand elasticities, raising markups. The model combines that endogenous-market-power mechanism with externally disciplined heterogeneity in returns to scale. The hypothesis is that a given markup wedge is more distortionary when it is attached to a firm that can absorb large quantities with only a limited increase in cost.

The paper is designed as a structural welfare exercise. Returns to scale are disciplined using firm-level evidence together with gross-output production function estimates using the approach from #cite(<gandhi2020identification>, form: "prose") (henceforth GNR) in the spirit of #cite(<hubmer2025scalable>, form: "prose"), while firm capability is summarized by its marginal cost at a common anchor output, calibrated inside the model. In the quantitative implementation, firms are draws from a common continuous joint distribution $F(alpha, nu)$: the $alpha$ margin is externally disciplined, while the capability margin $nu$ has a Pareto marginal with tail index $xi$ and lower bound $underline(nu)$. The tail index $xi$ is calibrated; the lower bound $underline(nu)$ is a units-fixing scale solved by the baseline anchoring condition (below) rather than a moment-targeted parameter.

#v(0.5cm)
= The Model
#v(0.2cm)
== Households
#v(0.2cm)

*Household problem.* Time is discrete. There is a unit mass of identical households. The representative household consumes the final good $C_t$, supplies labor $L_t$, owns a diversified portfolio of firms, and rents capital services to producers at the competitive rental rate $R_t$. Firms also purchase the aggregate composite as materials. The household accumulates capital according to
#math.equation(block:true, numbering: none, $K_(t+1) = (1-delta_K) K_t + I_t,$)<eq:k_lom> The intertemporal first-order condition equates the marginal rate of substitution to the net return on capital, $U_C(C_t,L_t) = beta E_t [U_C (C_(t+1),L_(t+1))
(R_(t+1) + 1 - delta_K)]$, so in any steady state the rental rate is pinned by preferences and depreciation alone:
#math.equation(block:true, $R = 1/beta - (1 - delta_K).$)<eq:euler_R>
Equation @eq:euler_R holds in the market allocation and in the planner allocation alike: neither the markup wedge nor the allocation of capital across firms moves the household's required return. Capital is therefore supplied perfectly elastically at $R$, and the aggregate stock $K$ is a residual of firm demand rather than a constraint. The household maximizes lifetime utility
#math.equation(block:true, numbering: none, $E_0 sum_(t=0)^infinity beta^t U(C_t,L_t),$)
with period utility
#math.equation(block:true, numbering: none, $U(C_t,L_t) = ln C_t - chi frac(L_t^(1 + 1/phi), 1 + 1/phi), quad chi > 0, phi >= 0.$)<eq:utility>

#v(0.3cm)

*Aggregate composite.* The aggregate composite#footnote[*Notation.* Sector, market, and firm indices are written as subscripts: $Y_j$, $P_j$, $omega_j$, $A_j$, $n_j$ for sector-level objects, and $s_(j i)$, $mu_(j i)$, $y_(j i)$, $p_(j i)$ for firm $i$ in sector $j$. This matches the numerical implementation, where sectors are a finite simulated sample $j = 1, dots, J$ and aggregates are sums $sum_(j=1)^J$. In the underlying theory, however, sectors are a continuum $j in [0,1]$, so these objects are properly fields over the unit interval ($Y(j)$, $P(j)$, $omega(j)$, $s(j,i)$, and so on), with population aggregates given by integrals $integral_0^1 dot.c d j$ rather than sums. The subscript form is used purely for readability; it carries no discreteness assumption at the theory level.] is produced from a continuum of product markets $j in [0,1]$ with exposure weights $omega_j$:
#math.equation(block:true, $Q_t = [integral_0^1 omega_j^(1/eta) Y_(j,t)^((eta-1)/eta) d j]^(eta/(eta-1)), quad eta > 1,$)<eq:agg_ces>
with associated price index
#math.equation(block:true, $P_t = [integral_0^1 omega_j P_(j,t)^(1-eta) d j]^(1/(1-eta)).$)<eq:agg_price>
Gross composite output is split between household consumption and intermediate use:
#math.equation(block:true, $Q_t = C_t + M_t, quad M_t = integral_0^1 sum_(i in A_(j,t)) m_(j i,t) d j.$)<eq:resource>
The exposure weight $omega_j$ summarizes how much aggregate expenditure is exposed to product market $j$ and is normalized so that $integral_0^1 omega_j d j = 1$. In the current baseline the simulated sectors are pooled with normalized exposure weights so they all have the same weight. Materials are purchases of the same composite at price $P_t$. The expenditure allocated to market $j$ is $X_(j,t) equiv P_(j,t) Y_(j,t)$, and the corresponding demand schedule loads on total composite use:
#math.equation(block:true, numbering: none, $Y_(j,t) = omega_j (P_(j,t)/P_t)^(-eta) Q_t.$)<eq:sector_demand>

#v(0.5cm)
== Demand and Market Shares
#v(0.2cm)

*Nested demand structure.* There is a continuum of finite sectors indexed by $j in [0,1]$. Each sector is one product market. Let $X_(j,t)$ denote composite expenditure assigned to sector $j$. Within sector $j$ a finite number $n_(j,t)$ of firms compete. Firm $i in {1, ..., n_(j,t)}$ supplies quantity $y_(j i,t)$ and charges price $p_(j i,t)$. Conditional on sector expenditure, the within-sector composite good is CES. Throughout the baseline I write the aggregator and its price index with an explicit $1 slash n_(j,t)$ weight, where $n_(j,t) equiv |A_(j,t)|$ is the realized number of allocated producing firms (the set $A_(j,t)$ is defined in @sec:market_structure below):
#math.equation(block:true, $Y_(j,t) = [sum_(i in A_(j,t)) frac(1, n_(j,t)) y_(j i,t)^((gamma-1)/gamma)]^(gamma/(gamma-1)), quad gamma > eta,$)<eq:market_ces>
with price index
#math.equation(block:true, numbering: none, $P_(j,t) = [(frac(1, n_(j,t)))^gamma sum_(i in A_(j,t)) p_(j i,t)^(1-gamma)]^(1/(1-gamma)) = n_(j,t)^(gamma/(gamma-1)) [sum_(i in A_(j,t)) p_(j i,t)^(1-gamma)]^(1/(1-gamma)).$)<eq:market_price>
The exponent on the $1 slash n_(j,t)$ weight is the dual of the quantity aggregator @eq:market_ces: a common multiplicative weight $a$ on $y_(j i,t)^((gamma-1)/gamma)$ maps to $a^gamma$ inside the price index.
The within-market elasticity of substitution $gamma$ is common across markets: it governs how aggressively firms inside a market substitute against one another. The across-market elasticity $eta$ is a single global object governing substitution across product markets.

#v(0.3cm)

*Variety normalization.* The $1 slash n_(j,t)$ weight neutralizes mechanical love of variety. Under symmetry, $n$ firms each producing $y$ deliver an unweighted composite $n^(gamma/(gamma-1)) y$, so the bundle rises with the number of firms even at fixed per-firm output; the weight removes this term and yields $Y_(j,t) = y$, leaving only genuine reallocation and heterogeneity gains. It is otherwise a pure level adjustment: relative to the unweighted aggregator it rescales $Y_(j,t)$ and $P_(j,t)$ by reciprocal powers of $n_(j,t)$, leaving sector expenditure $X_(j,t) = P_(j,t) Y_(j,t)$ and every firm-level quantity, price, share, markup, and profit invariant, and the normalized bundle is what enters the top-level aggregator @eq:agg_ces. Only the within-sector nest carries this normalization. The exposure weights are fixed, so @eq:agg_ces and @eq:agg_price have no variety margin.

#v(0.3cm)

*Shares and markups.* The conditional demand for firm $i$ in market $j$ is therefore
#math.equation(block:true, $y_(j i,t) = (frac(1, n_(j,t)))^gamma (p_(j i,t)/P_(j,t))^(-gamma) Y_(j,t),$)<eq:firm_demand>
and its revenue share inside the market, in which the common $1 slash n_(j,t)$ weight cancels, is
#math.equation(block:true, $s_(j i,t) equiv frac(p_(j i,t) y_(j i,t), sum_h p_(j h,t) y_(j h,t)) = frac(p_(j i,t)^(1-gamma), sum_h p_(j h,t)^(1-gamma)).$)<eq:share_def>

Finite-firm competition implies that firms internalize the effect of their own quantity choices on the market price index. In the nested-CES Cournot environment studied here and by #cite(<edmond2023costly>, form: "prose") and by #cite(<atkeson2008pricing>, form: "prose"), this generates an inverse-markup rule of the form
#math.equation(block:true, $frac(1,mu_(j i,t)) = 1 - frac(1,gamma) - (frac(1,eta) - frac(1,gamma)) s_(j i,t).$)<eq:emx_markup>
Equation @eq:emx_markup is the key pricing object in the model. Because $gamma > eta$, the coefficient on $s_(j i,t)$ is negative, implying that firms with larger market shares face less elastic residual demand and charge higher markups. With a common $gamma$, the within-sector markup floor as $s_(j i,t) -> 0$ is the same across sectors, $mu_min = gamma/(gamma-1)$; all heterogeneity in realized markups comes from endogenous market shares, which in turn reflect the distribution of firm draws within each sector and the exogenous sector firm count. The detailed Cournot derivation is relegated to the appendix. The logical equilibrium implication from this framework is that markups and market shares are jointly determined inside each sector market.

#v(0.5cm)
== Firms and Technology
#v(0.2cm)

*Heterogeneity.* Firms differ along two dimensions. First, they differ in scalability, indexed by $alpha_(j i)$, which governs gross-output returns to scale and how fast marginal cost rises with output. Second, they differ in capability, indexed by $nu_(j i)$, the inverse-marginal-cost shifter that fixes each firm's marginal cost at a common anchor output $hat(y)$: a higher $nu_(j i)$ means a lower marginal cost at the anchor. The empirical strategy treats $alpha_(j i)$ as externally disciplined and treats $nu_(j i)$ as the capability primitive. The dependence between $alpha$ and $nu$ is introduced only through ranks. Let $tilde(alpha)_(j i)$ be the standard-normal score of firm $i$'s scalability rank in sector $j$, and let $epsilon_(j i) ~ N(0,1)$. The capability ladder is
#math.equation(block:true, $ell_(j i) = macron(rho) tilde(alpha)_(j i) + sqrt(1 - macron(rho)^2) epsilon_(j i),$)<eq:nu_ladder>
and the ladder percentile is mapped into a Pareto capability draw,
#math.equation(block:true, $u_(j i) = Phi(ell_(j i)), quad nu_(j i) = underline(nu) (1 - u_(j i))^(-1/xi), quad xi > 1.$)<eq:nu_pareto>
The Pareto tail index $xi$ is calibrated to concentration moments, the lower bound $underline(nu)$ is a units-fixing scale solved by the baseline anchoring condition (not moment-targeted), and the alignment of $nu$ with $alpha$ is set through the calibrated rank correlation $macron(rho)$, which fixes the strength of the $alpha$--$nu$ rank copula. It is identified by the raw empirical correlation between firm-level returns to scale $alpha$ and log sales, $+0.596$, a directly observed cross-sectional moment requiring no productivity proxy and no data-side level anchor (See in @sec:calibration). A productivity-based moment is avoided because, with firm-specific returns to scale, measured TFP has no common units across firms, the same issue #cite(<hubmer2025scalable>, form: "prose") face and resolve the same way#footnote[ See @sec:data for the full argument.].

#v(0.3cm)

*Technology and Anchoring.* Let $phi.alt_v$ denote the value-added weight. Production in firm $i$ of market $j$ is gross output, combining the value-added composite $v_(j i,t) equiv k_(j i,t)^(a) l_(j i,t)^(1-a)$ with materials $m_(j i,t)$ in a constant-returns Cobb-Douglas input bundle $x_(j i,t) equiv v_(j i,t)^(phi.alt_v) m_(j i,t)^(1-phi.alt_v)$:
#math.equation(block:true, $y_(j i,t) = z_(j i) x_(j i,t)^(alpha_(j i)), quad x_(j i,t) = v_(j i,t)^(phi.alt_v) m_(j i,t)^(1-phi.alt_v), quad v_(j i,t) = k_(j i,t)^(a) l_(j i,t)^(1-a), quad phi.alt_v in (0,1).$)<eq:production>
The parameter $a$ is the common capital share inside value added. The value-added weight $phi.alt_v$ and materials weight $1-phi.alt_v$ give gross-output elasticities $(partial ln y)/(partial ln m) = (1-phi.alt_v)alpha_(j i)$, $(partial ln y)/(partial ln k) = phi.alt_v a alpha_(j i)$, and $(partial ln y)/(partial ln l) = phi.alt_v (1-a) alpha_(j i)$, summing to $alpha_(j i)$. The anchored parameterization takes capability $nu_(j i)$ as the primitive and treats $z_(j i)$ as a derived object, with the two being linked by the relabeling identity
#math.equation(block:true, $z_(j i) = (nu_(j i) \/ alpha_(j i))^(alpha_(j i)) hat(y)^(1 - alpha_(j i)),$)<eq:relabel>
so @eq:production, the input bundle, and the conditional input demands are algebraically unchanged. Only the distributional assumption moves: the common Pareto is now asserted on the dimensionally coherent cost shifter $nu_(j i)$ (marginal cost at the common anchor $hat(y)$) rather than on $z_(j i)$, whose units differ across firms when $alpha_(j i)$ differs.

#v(0.3cm)

*Cost minimization.* Cost minimization in the inner value-added problem implies the common unit cost index
#math.equation(block:true, $Omega(W_t,R_t) = (R_t/a)^(a) (W_t/(1-a))^(1-a).$)<eq:omega>
With materials purchased at price $P_t$, the gross-output unit cost is
#math.equation(block:true, $Omega^g_t equiv Omega^g (W_t,R_t,P_t) = (Omega(W_t,R_t)/phi.alt_v)^(phi.alt_v) (P_t/(1-phi.alt_v))^(1-phi.alt_v).$)<eq:omega_gross>
Given the production technology in @eq:production, expressed through the capability primitive $nu_(j i)$ at the anchor $hat(y)$, variable cost can be written in anchored form as
#math.equation(block:true, $T C_(j i,t)(y) = alpha_(j i) frac(Omega^g_t, nu_(j i)) hat(y) (y/hat(y))^(1/alpha_(j i)),$)<eq:total_cost>
which delivers marginal cost
#math.equation(block:true, $M C_(j i,t)(y) = frac(Omega^g_t, nu_(j i)) (y/hat(y))^(1/alpha_(j i)-1).$)<eq:marginal_cost>
At the anchor $y = hat(y)$, marginal cost is $M C_(j i,t)(hat(y)) = Omega^g_t \/ nu_(j i)$ regardless of $alpha_(j i)$: capability alone ranks firms at the anchor, and $alpha_(j i)$ only tilts how marginal cost moves as output deviates from it, so the level channel and the curvature channel are separated by construction.#footnote[The anchored form is an algebraic identity of the microfoundation cost function, not a new term. Factoring $y^(1/alpha_(j i)) = hat(y)^(1/alpha_(j i)) (y/hat(y))^(1/alpha_(j i))$ in the original total cost $Omega^g_t (y/z_(j i))^(1/alpha_(j i))$ gives $[Omega^g_t (hat(y)/z_(j i))^(1/alpha_(j i))] (y/hat(y))^(1/alpha_(j i)) equiv alpha_(j i) m_(j i) hat(y) (y/hat(y))^(1/alpha_(j i))$, with $m_(j i) equiv M C_(j i,t)(hat(y)) = Omega^g_t \/ nu_(j i)$ a forced, not chosen, definition. This recasts the sole units-carrying constant of the Cobb-Douglas DRS cost function, the output unit hidden inside $z^(-1/alpha)$, as the dimensionally coherent $nu_(j i)$, on which the Pareto assumption is asserted.]
Cost minimization also gives the conditional input demands
#math.equation(block:true, $P_t m_(j i,t) = (1-phi.alt_v) T C_(j i,t), quad W_t l_(j i,t) + R_t k_(j i,t) = phi.alt_v T C_(j i,t), quad frac(k_(j i,t),l_(j i,t)) = frac(a,1-a) frac(W_t,R_t).$)<eq:input_demand>
Equation @eq:marginal_cost makes the role of scalability transparent. The scale term $(y/hat(y))^(1/alpha_(j i)-1)$ shows that a larger $alpha_(j i)$ flattens marginal cost with respect to output around the anchor and allows the firm to expand more aggressively before costs rise sharply.

#v(0.3cm)

*Pricing and profits.* Conditional on demand, the equilibrium pricing condition is
#math.equation(block:true, $p_(j i,t) = mu_(j i,t) M C_(j i,t)(y_(j i,t)).$)<eq:pricing>
Profits can then be written as
#math.equation(block:true, $d_(j i,t) = p_(j i,t) y_(j i,t) - T C_(j i,t)(y_(j i,t)) = (1 - alpha_(j i) / mu_(j i,t)) p_(j i,t) y_(j i,t).$)<eq:profits>
Together, @eq:firm_demand, @eq:emx_markup, and @eq:pricing form a fixed-point system inside each market: quantities determine shares, shares determine markups, and markups feed back into prices and quantities through marginal cost.

#v(0.5cm)
== Market Structure and Decentralized Equilibrium<sec:market_structure>
#v(0.2cm)

*Market Structure.* I follows the EMX oligopoly implementation closely. Each sector $j$ receives an exogenous number of producing firms,
#math.equation(block:true, $n_(j,t) = max(1, "Poisson"(N)),$)<eq:poisson_count>
where $N$ is a calibrated primitive and is interpreted as the mean firms per sector. Conditional on this draw, the set $A_(j,t) = {1, dots, n_(j,t)}$ contains the firms priced in the sector Cournot problem. All allocated firms produce in the static cross section. There is no per-period fixed production cost and no profitability-based removal of firms.

The pool size $H$ used in the code is only a numerical slot count: it must be large enough that the Poisson draw rarely reaches the available slots. The behavioral market-structure parameter is $N$, not $H$. The realized count $n_(j,t)$ enters the within-sector aggregator @eq:market_ces and is reported in calibration output as a diagnostic, but it is not itself a moment matched to an assigned target.

#v(0.3cm)

*Equilibrium.* Given the aggregate state $K_t$, the exogenous sector counts, and the distributions of $(alpha_(j i),nu_(j i))$, a decentralized equilibrium is a sequence of household allocations, firm allocations, prices, market shares, and sector firm sets such that:

#v(0.5cm)
1.  households optimize given prices and firm payouts;
#v(0.2cm)
2. firms satisfy demand @eq:firm_demand and markup pricing @eq:pricing;
#v(0.2cm)
3. each sector firm set $A_(j,t)$ is the exogenous Poisson allocation in @eq:poisson_count;
#v(0.2cm)
4. labor and capital services clear:
  #math.equation(block:true, $L_t = integral_0^1 sum_(i in A_(j,t)) l_(j i,t) d j,$)<eq:labor_market>
  #math.equation(block:true, $K_t = integral_0^1 sum_(i in A_(j,t)) k_(j i,t) d j,$)<eq:capital_market>
#v(0.2cm)
5. composite feasibility holds as in @eq:resource, with materials clearing through $M_t = integral_0^1 sum_(i in A_(j,t)) m_(j i,t) d j$ and the aggregate price level normalized by $P_t equiv 1$.
#v(0.5cm)

Aggregate accounting closes because gross sales equal $P_t Q_t$, variable cost equals factor payments plus materials purchases, and profits are the residual. Hence value added satisfies $P_t (Q_t-M_t) = P_t C_t = W_t L_t + R_t K_t + Pi_t$: materials net out as intra-economy purchases. The equilibrium already pins down markups endogenously through market shares and allocates production across firms with different scalability. It does not yet require the full quantitative closure of the aggregate capital block or the exact mapping from the latent $z$ distribution to observed concentration moments. Those steps belong to the steady-state analysis deferred below.

#v(0.5cm)
== Planner's Allocation and Welfare<sec:planner_welfare>
#v(0.2cm)

*Planner Problem.*  The planner faces the same technology, the same exogenous sector firm sets, and the same distributions of scalability and latent capability.
The difference is that the planner does not treat markup pricing as a technological constraint. Instead, the planner allocates labor, capital services,
materials, and output so as to maximize household welfare subject only to feasibility, the production technology, and the same allocated firms. Formally, conditional on the producing firm set $A_(j,t)$ in each market, the planner chooses sequences
#math.equation(block:true, numbering: none, ${C_t, Y_(j,t), y_(j i,t), l_(j i,t), k_(j i,t), m_(j i,t)}_(t>=0)$)
to maximize
#math.equation(block:true, numbering: none, $W = E_0 sum_(t=0)^infinity beta^t U(C_t,L_t),$)<eq:welfare_objective>
subject to the household aggregator @eq:agg_ces, the market bundles @eq:market_ces, the resource constraint @eq:resource, the production technology @eq:production, and the aggregate labor and capital constraints @eq:labor_market and @eq:capital_market. Let $lambda_t^L$ and $lambda_t^K$ denote the multipliers on the aggregate labor and capital constraints, and let
#math.equation(block:true, numbering: none, $psi_(j i,t) equiv (partial Q_t) / (partial y_(j i,t))$)<eq:planner_price>
denote the planner's marginal contribution of one additional unit of firm output to the gross composite bundle; since $C_t = Q_t - M_t$, this is also the marginal contribution to consumption holding materials fixed. The planner's allocation conditions are then
#math.equation(block:true, $U_C(C_t,L_t) psi_(j i,t) (partial y_(j i,t))/(partial l_(j i,t)) = lambda_t^L,$)<eq:planner_labor_foc>
#math.equation(block:true, $U_C(C_t,L_t) psi_(j i,t) (partial y_(j i,t))/(partial k_(j i,t)) = lambda_t^K.$)<eq:planner_capital_foc>
#math.equation(block:true, $U_C(C_t,L_t) psi_(j i,t) (partial y_(j i,t))/(partial m_(j i,t)) = U_C(C_t,L_t).$)<eq:planner_materials_foc>
The planner equalizes the social marginal value of labor and capital across product lines, accounting for heterogeneity in scalability and capability, and uses materials until their marginal product equals the value of the composite diverted from consumption. Because $psi_(j i,t)$ comes from the same demand system as the decentralized equilibrium, planner and market share a common inverse-demand structure and differ only in the markup wedge in pricing#footnote[@sec:planner_appendix gives $psi_(j i,t)$ explicitly and maps it through the nested CES structure.].

#v(0.3cm)

*Regime invariance and the welfare scale.* Every allocation considered below is solved under the same two normalizations, $P = 1$ and $L = 1$, and at the same
required return @eq:euler_R. Labor is therefore pinned across regimes while capital is free in both the market and the planner allocation: by
@eq:factor_aggregates, the aggregate capital stock is whatever firms demand at $R$. For any allocation $x$ solved under these normalizations, lifetime
welfare is
#math.equation(block:true, numbering: none, $W^x = E_0 sum_(t=0)^infinity beta^t [ln C_t^x - chi frac((L_t^x)^(1 + 1/phi), 1 + 1/phi)],$)<eq:welfare_def>
and the consumption-equivalent gain from moving from the market (ME) to the planner (PE)
allocation is the scalar $lambda$ solving
#math.equation(block:true, numbering: none, $E_0 sum_(t=0)^infinity beta^t [ln((1+lambda) C_t^(M E)) - chi frac((L_t^(M E))^(1 + 1/phi), 1 + 1/phi)] = W^(P E),$)<eq:ce_welfare>
so that, with log utility over consumption,
#math.equation(block:true, $lambda = exp((1-beta)(W^(P E) - W^(M E))) - 1.$)<eq:ce_closed_form>
The labor-disutility weight $chi$ is held at its baseline market value in every
regime. Since $L = 1$ in every regime, the disutility term is common and
@eq:ce_closed_form collapses exactly to
#math.equation(block:true, $1 + lambda = C^(P E) / C^(M E),$)<eq:lambda_is_C_ratio>
For the channel exercises below, I define the object $Delta equiv ln(1+lambda)$, which is additive across any chain of intermediate
allocations. The channel decompositions below are therefore exact identities,
not local approximations.

#v(0.3cm)

*Four allocations.* Let us define $Delta W_"markup" = W_"Planner" (A) - W_"Market" (A)$
for the baseline gap at fixed sector firm sets $A$#footnote[Because the static cross section has exogenous firm counts, the planner does not re-select firms, so the full EMX entry counterfactual is left outside the baseline welfare exercise. The rationale behind this omission is the EMX finding that the entry wedge is quantitatively small relative to the markup-level and misallocation channels.]. To decompose it, I insert two intermediate allocations between the market and the planner, each solved on
the same draws $(alpha_(j i), nu_(j i))$, the same active set $A$, and the same frozen anchor $hat(y)$:
#v(0.2cm)

+ *Market* ($M E$): markup pricing @eq:pricing with $mu_(j i)$ from
   @eq:emx_markup.

+ *Planner* ($P E$): $mu_(j i) equiv 1$, capital free at @eq:euler_R.

+ *Fixed-input planner* ($P E_I$): $mu_(j i) equiv 1$ with the aggregate
   capital stock pinned to its market value, $K = K^(M E)$. The required return
   is no longer the Euler rate: $R$ adjusts upward until aggregate capital
   demand is choked down to the market envelope. Because $R$ enters every
   firm's unit cost @eq:omega_gross, this genuinely re-prices the whole cross
   section rather than relabeling the planner allocation.

+ *Uniform-markup* ($U$): every firm charges the common pure markup
   $mu_(j i) equiv macron(mu)$, where $macron(mu) = mu_("cw")^("model")$ is the
   market equilibrium's cost-weighted markup @eq:mu_cw_identity. The aggregate
   markup wedge is preserved by construction; only its cross-firm dispersion is
   removed.

#v(0.3cm)

*Two lenses.* The four allocations above give two exact two-way splits of the same total. Lens A holds the primary input fixed and asks how much of the
loss is pure misallocation, in the spirit of #cite(<hsieh2009misallocation>, form: "prose"):
#math.equation(block:true, $1 + lambda_"realloc" = C^(P E_I)/C^(M E), quad 1 + lambda_"scale" = C^(P E)/C^(P E_I).$)<eq:lens_a>
Lens B holds the aggregate markup level fixed and asks how much of the loss is dispersion rather than level, in the spirit of #cite(<edmond2023costly>, form: "prose"):
#math.equation(block:true, $1 + lambda_"disp" = C^U/C^(M E), quad 1 + lambda_"level" = C^(P E)/C^U.$)<eq:lens_b>
Both splits telescope, so
#math.equation(block:true, $(1+lambda_"realloc")(1+lambda_"scale") = (1+lambda_"disp")(1+lambda_"level") = 1 + lambda_"total",$)<eq:lens_identity>
or equivalently, in logs,
#math.equation(block:true, $Delta_"realloc" + Delta_"scale" = Delta_"disp" + Delta_"level" = Delta_"total",$)<eq:lens_identity_log>
with $lambda_"total"$ the object in @eq:ce_closed_form.
The two lenses are not nested: Lens A cuts the loss by which inputs move, Lens B by which wedges
move. Because all four legs are chained through @eq:lambda_is_C_ratio, the additive shares $Delta_"realloc" slash Delta_"total"$ and
$Delta_"disp" slash Delta_"total"$ are well defined and sum to one within each lens.
#v(0.5cm)
= Quantification
#v(0.2cm)
== Data and Measurement<sec:data>
#v(0.2cm)

*Sample and Measurement.* In the empirical pipeline, the baseline window is 2010--2019, and the NAICS2 data used to discipline calibration moments exclude sectors 52/53/62/81 following #cite(form: "prose", <hubmer2025scalable>), 92/99 for being out of scope, and 49 for having 8-11 firms and mean markup smaller than 1. This leaves a final total of 14 NAICS2 sectors. Concentration moments are measured within sector and then sector-sales-weighted across sectors, following EMX. Empirically, within each year-sector cell, firms are ranked by sales, where CR4 is the cumulative share at rank $min(3,n-1)$, CR20 at $min(19,n-1)$. These cell moments are time-averaged and collapsed using total sales weights. EMX's empirical reference is within 4-digit sectors. This project currently uses NAICS2 level of granularity due to Compustat exhibiting underpopulated sectors already at the NAICS3 level and beyond for concentration analysis.

Firm-level markups follow the production-function approach of #cite(<de2020rise>, form: "prose"): the ratio of price to marginal cost is the product of an estimated output elasticity and the firm's sales-to-cost ratio, aggregated across firms with cost weights and averaged over the window. Two variants of the same object are available, differing in the output elasticity rather than in the cost base: a COGS-only markup and an SG&A-inclusive one, which treats part of overhead as a variable input and is therefore lower, $1.185$ against $1.440$ in the retained sample. The SG&A-inclusive variant is the target used here, because the static Cournot problem has no separate overhead bucket and $T C_(j i)$ in @eq:total_cost is the firm's entire variable cost, so the COGS-only object would hold the model markup against a narrower empirical cost base than the model itself uses.

Following EMX, I compute the Autor-style long-difference estimate $hat(b)$, $Delta (1 slash mu_i) ~ Delta "HHI"_i$, across sectors. However, I regress the inverse cost-weighted markup directly on HHI, rather than following the labor-share-on-HHI regression used in EMX's oligopoly replication. The two recover the same structural slope: in this model class the sectoral labor share is proportional to $1 slash mu$, so a labor-share-on-HHI slope equals the inverse-markup-on-HHI slope up to a labor-share loading, and both identify $b = -(1 slash eta - 1 slash gamma)$. Using the markup directly removes that loading and, more importantly, sidesteps a measurement problem: in our Compustat panel labor cost (staff expense, `xlr`) is reported for only about one fifth of firm-years and is skewed toward large firms, which would bias any sectoral labor-share measure, whereas the markup is observed for the full sample#footnote[The choice of empirical specification is not material: re-estimating on our panel, the slope is about $-0.57$ in levels versus $-0.51$ in long differences, so the static cross-section used on the model side is comparable to the long-difference data target.].

#v(0.3cm)

*Scalability distribution.* The distribution $F^alpha$ is not calibrated, it is built once in the empirical pipeline and the solver samples draws from its support. The firm-level object is a hybrid GNR-KLEMS gross-output returns-to-scale estimate,
#math.equation(block:true,numbering: none, $alpha_(j i) = epsilon^m_(j i) + epsilon^k_(j i) + epsilon^k_(j i) frac(1 - a_s, a_s),$)
where $epsilon^m$ and $epsilon^k$ are the GNR gross-output materials and capital elasticities from the firm-year panel of #cite(<hubmer2025scalable>, form: "prose") and $a_s$ is the KLEMS value-added capital share of the firm's sector. The third term is an imputed labor elasticity, replacing the estimated GNR one with the elasticity implied by the KLEMS capital share, so that $epsilon^k slash (epsilon^k + epsilon^l) = a_s$ holds by construction#footnote[ The same object delivers the value-added weight $phi.alt_v = 1 - macron(epsilon)^m slash macron(alpha) = 0.348$ of @tab:param_ss.]. The imputation is needed because Compustat measures labor only as headcount. With employment as the labor input, the estimated GNR labor elasticity falls steeply with firm size (correlation $-0.92$ with log sales) and turns the returns-to-scale gradient negative ($-0.25$), although the materials and capital elasticities alone rise with size ($+0.80$). Wage-bill alternatives built from the sparse, size-skewed staff-expense series discussed above suffer from the same or worse problems, so a firm-level labor elasticity is not recoverable from Compustat.

Firm-years are collapsed to one window mean per firm, so $F^alpha$ is a cross section over permanent scalability types, pooled across retained sectors rather than binned by NAICS2. The pooled distribution is winsorized at the 1st and 99th percentiles, discretized to 500 equal-mass nodes, shrunk toward its own mean as $alpha^"shrunk" = macron(alpha) + lambda (alpha - macron(alpha))$ with $lambda = 0.371$, and clipped to the admissible support $[0.60, 1.20]$#footnote[The lower clip is a conservative admissible floor justified by the lowest industry-average returns to scale #cite(<hubmer2025scalable>, form: "prose") report (0.59, healthcare), and binds for 3.6% of the support mass. The upper clip never binds.]. The shrink is externally disciplined rather than chosen: the raw KLEMS-imputed dispersion is implausibly wide for a production elasticity (standard deviation $0.267$, 99th percentile $1.55$), and $lambda = 0.371$ brings the 99th percentile to $1.080$, reproducing the above-one upper tail of #cite(<hubmer2025scalable>, form: "prose"). The resulting model-facing support has mean $0.806$, standard deviation $0.094$, and 3.4% of its mass at $alpha >= 1$#footnote[ @fig:alpha_support in the appendix shows the raw, winsorized, and clipped distributions.]. An ideal future scalability distribution $F^alpha$ construction would use the same data as #cite(<hubmer2025scalable>, form: "prose") from U.S Census in order to avoid running into this issue.

#v(0.3cm)

*Sorting Evidence.* @fig:alpha_by_size plots mean scalability against the firm's sales percentile, in twenty equal-width bins, for the firm cross section behind the $alpha$--log sales moment of @sec:calibration ($6,230$ firms, hybrid $alpha$ before the shrink of $F^alpha$). Scalability rises with size throughout: firms in the bottom 5% of their sector's sales distribution have $alpha$ about $0.35$ below the mean and firms in the top decile about $0.18$ above it, a gap of nearly two standard deviations of raw $alpha$. The profile is concave, steep among small firms and flattening in the upper tail, and ranking firms within sector or in the pooled sample gives nearly the same curve, so the gradient is not a composition effect across sectors. In the model, $alpha$ alone does not deliver this ordering: at the anchor $hat(y)$ marginal cost is independent of $alpha$, and a higher $alpha$ lowers marginal cost above the anchor but raises it below, so scalability by itself spreads firms around $hat(y)$ rather than ranking them by size. A monotone gradient requires the most scalable firms to also be the most capable, which is the alignment $macron(rho)$ governs in @eq:nu_ladder and that the permutation counterfactuals of @sec:welfare take apart.

#figure(
  block(width: 70%)[
    #image("figures/fig_alpha_by_sales_percentile.pdf", width: 100%)
    #align(left, text(size: 7.5pt)[_Note._ Demeaned mean $alpha$ within twenty equal-width bins of the firm's sales percentile, with firms ranked either in the pooled sample or within their sector, for the $6,230$ firms behind the $alpha$--log sales moment of @sec:calibration, using the hybrid $alpha$ before the shrink of $F^alpha$.])
    #v(0.5em)
  ],
  caption: [*Mean scalability by sales percentile.*],
)<fig:alpha_by_size>

A more direct check would correlate $alpha$ with log TFPQ $ln hat(z)$ from the production-function estimates of #cite(<gandhi2020identification>, form: "prose"). That correlation is not informative here. The estimate is a residual against a common industry technology, whereas in the model each firm operates its own technology with returns $alpha_(j i)$, as in @eq:relabel. Read that way, TFPQ has no common units: rescaling output and inputs by a factor $s$ shifts $ln hat(z)_(j i)$ by $(1 - alpha_(j i)) ln s$, a firm-specific amount that moves with $alpha$ itself, so relative TFP across firms has no cardinal interpretation #cite(<hubmer2025scalable>). Across rescalings from $s = 10^(-3)$ to $10^3$, the within-sector rank correlation of $alpha$ and $ln hat(z)$ goes from $+0.27$ to $-0.08$. #cite(<hubmer2025scalable>, form: "prose") meet the same problem in their quantitative model and do not impose the estimated joint distribution of returns to scale and TFP. They calibrate the correlation between the two internally, disciplined by the gap in average returns to scale between the top 5% and the bottom half of firms by revenue.

#v(0.5cm)
== Pooled-Market Steady State<sec:steady_state>
#v(0.2cm)

I now characterize the steady state as a nested system. Time subscripts are dropped, the aggregate price index is normalized to $P = 1$, and the distorted market steady state is written under the labor normalization $L = 1$. The resulting equilibrium is then used to pin down the labor-disutility weight $chi$, after which the market and planner steady states can be compared in common units. The steady-state analysis is deliberately written at the generic sector-market level, as is common in the literature.

#v(0.3cm)

*Static inner-market block.* Conditional on sector $j$, the exogenous firm count $n_j$, and the realized distribution of firm draws $(alpha_(j i), nu_(j i))_(i in A_j)$, the steady-state market equilibrium solves for the tuple ${(y_(j i), p_(j i), s_(j i), mu_(j i), d_(j i), m_(j i))}_i$. The fixed point is generated by the demand system, the share-based markup rule, pricing over marginal cost, profits, and the conditional input demands:
#math.equation(block:true, numbering: none, $y_(j i) = (frac(1, n_j))^gamma (p_(j i)/P_j)^(-gamma) Y_j,$)
#math.equation(block:true, numbering: none, $frac(1,mu_(j i)) = 1 - frac(1,gamma) - (frac(1,eta) - frac(1,gamma)) s_(j i),$)
#math.equation(block:true, numbering: none, $p_(j i) = mu_(j i) M C_(j i)(y_(j i)),$)
#math.equation(block:true, numbering: none, $d_(j i) = p_(j i) y_(j i) - T C_(j i)(y_(j i)) = (1 - alpha_(j i) / mu_(j i)) p_(j i) y_(j i).$)
#math.equation(block:true, numbering: none, $P m_(j i) = (1-phi.alt_v) T C_(j i), quad w l_(j i) + R k_(j i) = phi.alt_v T C_(j i).$)
At this stage the object is entirely symbolic: the cross section of firms inside the sector is summarized by $(alpha, nu)$ draws and by the endogenous interaction between shares and markups. Quantitatively, the implementation draws a slot pool of size $H$, allocates $n_j = max(1, "Poisson"(N))$ producing firms, and prices exactly those firms. The $alpha$ draw comes from $F^alpha$ and the capability draw $nu$ follows @eq:nu_ladder and @eq:nu_pareto.

#v(0.3cm)

*Pooled aggregation block.* Sector outcomes aggregate directly over the continuum of product markets using exposure weights. Gross output is the exposure-weighted CES bundle $Q = [integral_0^1 omega_j^(1/eta) Y_j^((eta-1)/eta) d j]^(eta/(eta-1))$, with sector $j$ contributing in proportion to its exposure weight $omega_j$, and consumption is net output $C = Q - M$. In the quantitative implemention, I assume that all sectors cary the same $omega_j = omega, thick forall j$ for simplicity. Following the aggregation logic of #cite(<edmond2023costly>, form: "prose"), the aggregate markup relevant for the overall scale of production is the cost-weighted markup wedge, not the simple sales-weighted average of firm markups. The sales-weighted average loads both on the level of markups and on the composition of expenditure across firms, so it mixes the aggregate wedge with the dispersion component that drives misallocation.

The same aggregation logic is what makes the steady-state cross section economically informative. Firm-level markup dispersion changes the allocation of labor, capital, and materials across firms with different market shares and different scalability, thereby affecting aggregate productivity. Concentration matters not just because it shifts average markups, but because it changes which firms absorb resources and how fast their marginal costs rise.

#v(0.3cm)

*Capital and factor-market block.* Aggregating the conditional input demands @eq:input_demand over the continuum of markets gives aggregate factor use as a function of aggregate variable cost $T C equiv integral_0^1 sum_(i in A_j) T C_(j i) d j$ alone. Only the fraction $phi.alt_v$ of variable cost is a primary-factor payment; the remaining $1-phi.alt_v$ buys materials of the
composite at $P = 1$:
#math.equation(block:true, $K = frac(a phi.alt_v "TC", R), quad L = frac((1-a) phi.alt_v "TC", w), quad M = (1-phi.alt_v) "TC", quad C = Q - M.$)<eq:factor_aggregates>
With $R$ fixed at @eq:euler_R, @eq:factor_aggregates makes the aggregate capital stock a residual: whatever quantity firms demand at the Euler rate is supplied. The steady-state equilibrium is therefore a two-dimensional fixed point in $(w, X)$, where $X$ is composite expenditure per market, with the two residuals being the price normalization $P = 1$ and the labor normalization
$L = 1$. The rental rate is not among the unknowns.

#v(0.3cm)

*Normalization and steady-state welfare block.* The distorted market steady state is anchored by the normalizations $P = 1$ and $L = 1$, together with the baseline anchoring condition that the median active-firm output equals the common anchor, $tilde(y) = hat(y) equiv 1$, imposed as one equilibrium condition of the baseline market solve#footnote[This condition pins the capability scale $underline(nu)$ (the Pareto lower bound) as aby product of its level normalization.]. Conditional on assigned parameters and on the anchored capability distribution, the market solve pins down the wage-consumption ratio and therefore the labor-disutility weight
#math.equation(block:true, numbering: none, $chi = w / C,$)
using the market labor-leisure condition under $L = 1$ and $C = Q - M$. Once this baseline calibration step is complete, the planner and market steady states are re-solved in common units while holding the primitive objects fixed. The planner and market allocations therefore share the same $(beta, a, phi.alt_v, hat(y), alpha_(j i), nu_(j i), N, xi, H)$ and differ only through the presence or absence of the aggregate-markup and dispersion distortions. This is what makes the later welfare comparison interpretable as a comparison across allocations rather than across calibrations.

#v(0.3cm)

*Frozen-anchor protocol.* The anchor $hat(y)$ and all firm draws $(alpha_(j i), nu_(j i))$ are fixed at the baseline and held frozen across the planner solve, the uniform-markup decomposition, and every counterfactual: technology is then an ordinary allocation-independent object, so the planner faces literally the same $T C_(j i,t)(y)$ as the market and re-anchoring under the planner is structurally impossible. This follows the logic of the normalized-CES literature #cite(<klump2000economic>), in which technology families are normalized to coincide at a common benchmark before varying curvature parameters. Our fixed-$hat(y)$ construction applies the same identification principle to heterogeneous returns to scale: all technologies agree on marginal cost at the reference output, so differences in $alpha$ affect only the curvature of marginal cost away from that operating point. I go beyond #cite(<klump2000economic>, form: "prose") by arguing that the benchmark must also be economically sensible, and the median is preferable to the sales-weighted mean in skewed distributions#footnote[See @anchor for a discussion of the difference.].

#v(0.5cm)
== Calibration<sec:calibration>
#v(0.2cm)

*Calibrated parameters.* In the pooled baseline the calibrated vector is $(xi, N, gamma, eta, macron(rho))$. The within-market elasticity $gamma$ controls the common substitution environment and helps pin the markup level through the SG&A-inclusive aggregate cost-weighted markup target $mu_("cw,sga")^("data") approx 1.18$. The across-market elasticity $eta$ governs cross-sector substitution and is disciplined by the EMX slope moment. The Pareto tail index $xi$ and the Poisson mean $N$ jointly discipline concentration, especially CR4 and CR20: $xi$ is now the tail of the cost shifter $nu$ (approximately $z^(1/alpha)$), which maps directly into the sales and concentration tail, while $N$ sets the mean number of firms in a sector.

The rank correlation $macron(rho)$ sets the strength of the $alpha$--$nu$ rank copula and is calibrated to the raw empirical correlation between $alpha$ and log sales, $+0.596$. Its sign is not at odds with the negative returns-to-scale--TFP correlation, $-0.253$, calibrated by #cite(<hubmer2025scalable>, form: "prose"). Theirs is defined on TFP $z$, which by @eq:relabel loads on $alpha$ directly, whereas $macron(rho)$ is defined on the anchored capability $nu$. The capability scale $underline(nu)$ is not part of the calibrated vector: it is solved by the baseline anchoring condition $tilde(y) = hat(y) equiv 1$, which centers the $alpha$-gradient at the operating scale rather than fighting the correlation moment. Objects treated as externally disciplined are the pooled scalability distribution $F^alpha$, the firm-level gross-output returns to scale $alpha_(j i)$, the value-added weight $phi.alt_v$, the common capital share $a$, and the exposure distribution $omega_j$. See @tab:param_ss for an overview of all parameters and their final calibration values.


#v(0.5cm)

*Calibration targets.* The calibration targets are pooled scalars. Following EMX, the targets remain the SG&A-inclusive aggregate cost-weighted markup $mu_("cw,sga")^("data")$, CR4, CR20, and the EMX slope moment. Additionally, the raw correlation between $alpha$ and log sales, $+0.596$, is added as a primary target identifying $macron(rho)$. The model analogue of the markup target is #math.equation(numbering:none,block:true,$mu_("cw")^("model") = integral_0^1 sum_(i in A_j) lambda^c_(j i) mu_(j i) d j, quad lambda^c_(j i) equiv (T C_(j i))/(integral_0^1 sum_(h in A_j) T C_(j h) d j)$)
, where $lambda^c_(j i)$ is firm $i$'s share of aggregate variable cost. With heterogeneous $alpha_(j i) < 1$, the reliable welfare cost is the equilibrium gap $Delta W_("markup")$ from the planner-vs-market solve, which the roundabout channel amplifies and markup dispersion across heterogeneous $alpha_(j i)$ further shapes.

The model counterpart of the empirical $hat(b)$ target is the static sector-level regression of inverse cost-weighted markup on sector HHI, because model sectors have no persistent fixed effects. The cross-sector HHI variation this slope is identified off comes entirely from the Poisson draw in @eq:poisson_count: realized counts $n_j = |A_j|$ differ across simulated sectors, and sectors that happen to draw few firms are mechanically more concentrated.

#v(0.5cm)

*Calibration Results.* As shown in @tab:calib_fit, all five moment targets are hit to $tilde 0.15%$, with the worst fit being for the $"CR"20$ target, around $0.143%$ of. As for the resulting parameter values, the within-market elasticity $gamma$ comes at a distinctly low value when compared with other literature such as EMX. This has clear distributional impacts in that it lifts the markup floor to $1.160$ compared to EMX's $1.085$. In fact, $86.6%$ of the average markup wedge is the common CES floor, and only $tilde 13 %$ comes from endogenous share-driven market power. This is a mechanical consequence of the data: Compustat is public firms only, so measured concentration is low and shares are small, leading to a small share-driven component of markups. To match this empirical setup, $gamma$ goes low to make sure that the markup floor will cary most of the markup level.

#v(0.3cm)
#figure(
  block(width: 96%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (2.5fr, 0.8fr, 2.3fr, 0.95fr, 0.95fr, 0.95fr),
      column-gutter: 0.30cm,
      row-gutter: 0.12cm,
      align: (left, right, left, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.header[*Parameter*][*Value*][*Target moment*][*Data*][*Model*][*Dev. (%)*],
      table.hline(stroke: 0.4pt),

      [$gamma$ (within-market elast.)], [7.25], [$mu_("cw,sga")$], [1.1848], [1.1847], [0.004],
      [$eta$ (across-market elast.)], [1.544], [Autor slope $hat(b)$], [$-0.5096$], [$-0.5096$], [0.000],
      [$xi$ (Pareto tail of $nu$)], [16.80], [CR4], [0.2035], [0.2034], [0.020],
      [$N$ (Poisson mean firms)], [57], [CR20], [0.4835], [0.4828], [0.143],
      [$macron(rho)$ (rank copula $alpha$--$nu$)], [0.897], [corr($alpha$, log sales)], [0.5957], [0.5960], [0.056],

      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ The five parameters are solved jointly, so the row pairing reports the moment each is most informative for rather than a sequential mapping. Dev. is the absolute deviation of the model moment from its target, in percent of the target. Full parameterization in Appendix @tab:param_ss.])
    #v(0.5em)
  ],
  caption: [*Calibrated parameters and model fit.*],
)<tab:calib_fit>
#v(0.3cm)


As for the across-sector elasticity $eta$, the Autor slope $hat(b)$ is an exact function of the two elasticities, so given $gamma$, matching $hat(b)$ inverts directly for $eta$, hence the 0% deviation. Moving on to the Pareto tail parameter $xi = 16.80$, it might look extreme compared to usual values found in the literature. However, note that it is a tail on capability $nu prop z^(1/alpha)$, and not productivity $z$ directly. In fact, since $alpha$ is heterogeneous, the implied $z$ distribution would be a mixture of Paretos with indices $xi "/" alpha_(j i)$, and therefore have no single tail index. This is precisely why the Pareto tail is asserted on the dimensionally coherent $nu$, and not directly comparable with tail parameters on productivity under constant returns to scale.

The Poisson mean $N$ also plays an important role together with $xi$ in matching the concentration targets. The observed CR4 and CR20 can come either from few competitors or dispersed capability, and the resulting $N = 57$ and fit tells us that it lands decisively on the second: With 57 near-symmetric firms the mechanical top-4 share would be $4 slash 57 approx 7%$; the calibrated $nu$ tail through $xi$ nearly triples it to the $20.3%$ target. Therefore, concentration in the model is very much a capability-dispersion phenomenon, not a small-market phenomenon. The distinction matters for @sec:welfare: it is the upper tail of $nu$ that determines which firms grow large and hence which firms earn high markups, so once $nu$ is strongly sorted with $alpha$ the largest wedges land on the most scalable firms rather than on an arbitrary subset

Finally, the rank copula of $macron(rho) = 0.897$ is not the direct correlation object between capability and scalability. In fact, it maps onto corr$(nu,alpha) approx 0.785$ through the nonlinear Pareto transform. This strong positive assortativity is central to the welfare exercise done in @sec:welfare, and, more importantly, disciplined by an observable instead of assumed.


#v(0.5cm)
= Welfare Analysis<sec:welfare>

#v(0.5cm)

== Markup Distribution<sec:markup_dist>
#v(0.2cm)

*Compression relative to EMX.* I compare the distribution of markups in the calibrated economy presented here versus that of EMX since their framework is the closest reference point, despite the lack of heterogeneous returns-to-scale parameters, as discussed below.
The calibrated market economy generates a markup distribution that is far more compressed than that of EMX, both across sectors and across firms, as can be seen in @tab:markup_dist. At the firm level, the cost-weighted interquartile range goes from $1.170$ to $1.176$, against $1.09$ to $1.17$ in EMX. Additionally, the 99th percentile sits $0.17$ above the median, whereas the same gap in EMX is $0.46$, almost three times bigger.

From below, the compression comes from the low calibrated $gamma$, which lifts the common CES floor to $1.160$ and carries most of the markup level for the reasons discussed in @sec:calibration. From above, the markup behaves as a monotone increasing function of the market share by @eq:emx_markup, and the low concentration of the Compustat targets keeps the largest shares in check. Decreasing returns to scale are also potentially compressing the tail further, since with most of the mass of $F^alpha$ below one, marginal cost @eq:marginal_cost rises above the anchor at the rate $1 slash alpha_(j i) - 1$, a convex brake that bites the hardest on the largest firms#footnote[I have not isolated this channel with a constant-returns counterfactual, however, so it should be taken as a conjecture rather than as a result.].


#v(0.3cm)
#figure(
  block(width: 80%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr),
      column-gutter: 0.30cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      [], table.cell(colspan: 2, align: center)[*This paper*], table.cell(colspan: 2, align: center)[*EMX* ($M = 1.15$)],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*Percentile*], [Sector $mu_j$], [Firm $mu_(j i)$], [Sector $mu_j$], [Firm $mu_(j i)$],
      table.hline(stroke: 0.4pt),

      [25th], [1.174], [1.170], [1.12], [1.09],
      [50th], [1.179], [1.171], [1.14], [1.11],
      [75th], [1.189], [1.176], [1.16], [1.17],
      [90th], [1.203], [1.220], [1.21], [1.27],
      [99th], [1.238], [1.345], [1.35], [1.57],

      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ Sector $mu_j$ is the cost-weighted mean of firm markups within sector $j$, with percentiles weighted by sector cost. Firm percentiles are weighted by firm cost.])
    #v(0.5em)
  ],
  caption: [*Cost-weighted distribution of sector and firm markups, baseline market against EMX.*],
)<tab:markup_dist>
#v(0.3cm)

*Dispersion.* The dispersion that does survive the three identified contraction forces is concentrated in the upper tail of the firm distribution. The firm percentiles lie below the sector ones up to the 75th percentile and above them from the 90th onwards, with the firm 99th percentile at $1.345$ against $1.238$ across sectors. By @eq:emx_markup, this 99th percentile corresponds to a firm holding about $23%$ of its sector's sales, whereas the median firm holds less than $2%$. Therefore, within-sector dispersion comes from a few dominant firms per sector rather than from a considerable spread of pricing power, which has two consequences for the welfare exercise. The first is that any welfare loss from markup dispersion must come from this thin tail. The second is that the calibration disciplines the markup level and concentration but not markup dispersion, such that the tail is an untargeted object and the absolute size of any dispersion-driven loss should be taken with a grain of salt. The sorting comparisons of @sec:sorting are less exposed to it, since all arrangements share the same marginal distributions and the same random draws, so the tail moves the levels but not the differences.


#v(0.5cm)
== Welfare Cost of Markups<sec:welfare_baseline>
#v(0.2cm)

*Welfare Setup.* Using the framework described in @sec:planner_welfare, I decompose the welfare results into the two lenses A and B, and within each lens, I further break it down into reallocation and scale legs and  dispersion and level legs, respectively. The first lens (Lens A) and its decomposition aims at measuring the welfare impact of market power through a missallocation view, by holding inputs fixed a lá #cite(form: "prose", <hsieh2009misallocation>), while the second lens (Lens B) focuses on breaking down the welfare impact of the markup wedge into the dispersion wedge and the aggregate dispersion wedge, by holding the aggregate markup level fixed, similar to the exercise done by EMX. Later, I will explore the role of sorting between scalability and capability within each of these lenses and legs.

#v(0.5cm)

*Lens A.* Moving from the market to the planner allocation raises steady-state consumption from $60.63$ to $79.99$, which by @eq:lambda_is_C_ratio is a consumption-equivalent gain of $lambda_"total" = 31.9%$. Most of this gain comes from scale rather than from the reallocation of a given stock of inputs, as can be seen in Panel B of @tab:welfare_lenses under lens A: Holding the primary input at its market level, the fixed-input planner of @eq:lens_a recovers only $lambda_"realloc" = 5.8%$ ($20.2%$ of $Delta_"total"$), whereas letting capital adjust delivers the remaining $lambda_"scale" = 24.8%$ ($79.8%$ of $Delta_"total"$).

The scale leg is a capital-accumulation margin, since with the markup wedge removed and the return pinned at @eq:euler_R, the planner operates a capital stock $2.14$ times that of the market. Conversely, when capital is held at $K^(M E)$, the rental rate that clears the capital market under efficient pricing rises from $0.102$ to $0.166$, about $63%$ above the Euler rate, which helps us understand how far the markup wedge holds accumulation below its efficient level.

#v(0.3cm)
#figure(
  block(width: 90%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (2.3fr, 1.2fr, 1fr, 1fr, 1.2fr),
      column-gutter: 0.30cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.cell(colspan: 5, inset: (top: 6pt, bottom: 4pt))[*Panel A.* Allocations],
      table.hline(stroke: 0.4pt),
      [*Allocation*], [$mu_("cw")$], [$K$], [$R$], [$C$],
      [Market ($M E$)], [1.184], [109.4], [0.102], [60.63],
      [Fixed-input planner ($P E_I$)], [1.000], [109.4], [0.166], [64.12],
      [Uniform markup ($U$)], [1.184], [117.9], [0.102], [62.90],
      [Planner ($P E$)], [1.000], [234.2], [0.102], [79.99],
      table.hline(stroke: 0.7pt),

      table.cell(colspan: 5, inset: (top: 10pt, bottom: 4pt))[*Panel B.* Welfare legs],
      table.hline(stroke: 0.4pt),
      [*Leg*], [*Step*], [$lambda$], [$Delta$], [Share of $Delta_"total"$],
      table.cell(colspan: 5, inset: (top: 5pt, bottom: 2.5pt))[_Lens A: fixed input_],
      [#h(0.8em)Reallocation], [$M E -> P E_I$], [5.8%], [0.056], [20.2%],
      [#h(0.8em)Scale], [$P E_I -> P E$], [24.8%], [0.221], [79.8%],
      table.cell(colspan: 5, inset: (top: 5pt, bottom: 2.5pt))[_Lens B: uniform markup_],
      [#h(0.8em)Dispersion], [$M E -> U$], [3.7%], [0.037], [13.3%],
      [#h(0.8em)Level], [$U -> P E$], [27.2%], [0.240], [86.7%],
      table.hline(stroke: 0.4pt),
      table.cell(inset: (top: 4pt, bottom: 4pt))[Total],
      table.cell(inset: (top: 4pt, bottom: 4pt))[$M E -> P E$],
      table.cell(inset: (top: 4pt, bottom: 4pt))[31.9%],
      table.cell(inset: (top: 4pt, bottom: 4pt))[0.277],
      table.cell(inset: (top: 4pt, bottom: 4pt))[100%],
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ Panel A reports the four allocations that define the two lenses in @eq:lens_a and @eq:lens_b, all solved under $P = 1$ and $L = 1$. In Panel B, $lambda$ is the consumption-equivalent gain of each leg and $Delta = ln(1 + lambda)$ its additive counterpart.])
    #v(0.5em)
  ],
  caption: [*Baseline allocations and the two-lens decomposition of the welfare cost of markups.*],
)<tab:welfare_lenses>
#v(0.3cm)

*Lens B.* Replacing every firm markup with the common markup $macron(mu) = 1.184$ removes dispersion while preserving the aggregate wedge, and by @eq:lens_b it is worth only $lambda_"disp" = 3.7%$ ($13.3%$ of $Delta_"total"$), such that the markup level accounts for the remaining $lambda_"level" = 27.2%$ ($86.7%$). This is the magnitude that the compressed distribution of @sec:markup_dist anticipates, and also a common finding in the literature #cite(<edmond2023costly>) #cite(<de2020rise>) #cite(<arkolakis2010market>).

Moving one, one might read the two small legs as two estimates of the same misallocation object, but that would be an incorrect interpretation, since they cut the loss along different margins. The reallocation leg removes all wedges with capital and labor fixed, whereas the dispersion leg keeps the level wedge but lets capital respond, and capital in the uniform-markup economy is $7.8%$ higher than in the market. Both lenses therefore place most of the loss on the aggregate scale of production and a fifth or less on the reallocation and dispersion legs, which leaves open what role the sorting of scalability onto capability plays in each of the four legs, the question I turn to in @sec:sorting.

#v(0.5cm)
== Sorting and the Cost of Markups<sec:sorting>
#v(0.2cm)

*The $alpha$-Arrangement Experiment.* The welfare cost of @sec:welfare_baseline is measured in an economy where scalability and capability are strongly aligned, with corr$(alpha, nu) = 0.785$ across active firms, the model counterpart of the calibrated copula discussed in @sec:calibration. To isolate the contribution of this alignment, I re-solve the allocations of @sec:planner_welfare under four arrangements of $alpha$ across firms, all on the same draws of $nu_(j i)$, the same active set $A$, the same anchor $hat(y)$, and with $chi$ held at its baseline value. The first is the baseline, which keeps the calibrated positive sorting. The second is the shuffled arrangement, which randomly permutes $alpha$ among the active firms of each sector and brings corr$(alpha, nu) approx 0$, with every result averaged over ten permutation draws. The third is the reverse-sorted arrangement, which pairs the largest $alpha$ with the smallest $nu$ within each sector, yielding corr$(alpha, nu) = -0.82$. Since these four arrangements only permute $alpha$ within sectors, both marginal distributions are identical across them, and any difference in welfare comes from the pairing alone.. Finally, the homogeneous-$alpha$ arrangement replaces every $alpha_(j i)$ with its sector's harmonic mean, which preserves each sector's mean $1 slash alpha$, and hence its mean marginal-cost elasticity in @eq:marginal_cost, but removes the within-sector heterogeneity in scalability altogether.

Combining the homogeneous, shuffled and baseline arrangements splits any welfare leg additively on the $Delta$ scale,
#math.equation(block:true, $Delta_"base" = underbrace(Delta_"hom","common"-alpha "loss") + underbrace((Delta_"shuf" - Delta_"hom"), "heterogeneity term") + underbrace((Delta_"base" - Delta_"shuf"), "sorting term"),$)<eq:sorting_split>
where the first term is the common-$alpha$ loss, corresponding to a standard DRS framework, the second is the heterogeneity term, which adds dispersion in $alpha$ but assigns it at random, and the third is the sorting term, which assigns the most scalable technologies to the most capable firms following the baseline $macron(rho)$. Since every $Delta$ is a log consumption ratio by @eq:lambda_is_C_ratio, @eq:sorting_split holds exactly for $Delta_"total"$ and for each of the four legs in @eq:lens_identity_log.  The results of each experiment are displayed at lenght in @tab:sorting below.

#v(0.5cm)

*Lens A Decomposition: Reallocation $times$ Scale.* Positive sorting nearly doubles the welfare cost of markups on the $lambda$ scale, raising $lambda_"total"$ from $17.4%$ under the shuffled arrangement to $31.9%$ under the baseline (standard error of $0.012$ percentage points across the ten draws), as can be seen in Panel B of @tab:sorting. In comparison, the reverse sorting and homogenous $alpha$ lower the cost further, to $15.0%$ and $12.8%$ respectively, but not nearly as much as killing positive sorting does. It The welfare cost falls monotonicaly with $"corr"(alpha,nu)$, but the relationship is far from linear, as I show later below. The amplification is also proportional across the two legs of Lens A. By @eq:sorting_split, the sorting term accounts for $46.6%$ of $Delta_"realloc"$, $41.0%$ of $Delta_"scale"$ and $42.1%$ of $Delta_"total"$, whereas the common-$alpha$ term holds between $42.9%$ and $43.8%$ of each and heterogeneity assigned at random adds only $10.5%$ to $15.3%$, as reported in Panel C#footnote[The sorting term $Delta_"base" - Delta_"shuf"$ does not involve the homogeneous-$alpha$ arrangement, so the choice of variant only moves the line between the common-$alpha$ and heterogeneity terms. I also do a cost-weighted robustness, which assigns a single economy-wide $alpha$ that preserves the aggregate marginal-cost elasticity, $lambda_"total" = 16.2%$ and the common-$alpha$ term of $Delta_"scale"$ rises from $0.097$ to $0.122$.]. Therefore, the reallocation leg stays close to around one fifth of $Delta_"total"$ in every arrangement, since sorting scales both legs by similar proportions.

In levels, however, most of the sorting term sits in the scale lef, with $0.0906$ of the $0.1167$ on the $Delta$ scale as visible in Panel C, since sorting raises the planner's capital stock from $1.70$ to $2.14$ times that of the market (Panel A). Under positive sorting, the largest firm of each sector combines the highest $alpha$ and $nu$ with the highest markup, and the planner raises its inputs $5.6$-fold against $1.6$-fold for the bulk of firms, such that these leaders, only $1.8%$ of active firms, absorb $47.5%$ of the additional inputs. Since their markup exceeds that of the rest by only $7.5$ log points, the wedge acts as a trigger rather than the source of this expansion, which comes instead from the strong output response of their scalable technologies, with $nu$ adding no predictive power once $alpha$ and the markup are accounted for.#footnote[This comes from a descriptive cross-firm regression of $ln(T C^(P E) slash T C^(M E))$ on $ln mu^(M E)$, $alpha$ and $ln nu$ with sector fixed effects, where total cost measures firm inputs since every input is a fixed fraction of it within a regime. The partial $R^2$ is $0.63$ for the markup, $0.22$ for $alpha$ and $0.01$ for $ln nu$, whose coefficient is negative, although the markup is itself an equilibrium outcome of $alpha$ and $nu$, so the regression ranks predictors rather than identifying a channel.]

Note that reading this amplification as evidence that sorting is harmful would be an incorrect interpretation, since breaking sorting lowers consumption in every allocation of Panel A. Consumption falls from $60.63$ to $55.43$ in the market ($-8.6%$), from $64.12$ to $57.11$ in the fixed-input planner ($-10.9%$) and from $79.99$ to $65.08$ in the planner ($-18.6%$). Sorting is therefore productive, raising the cost of markups solely because the planner gains more from it than the market does, not because the absolute level falls (on the contrary).

#v(0.3cm)
#figure(
  block(width: 100%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (2.2fr, 1fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
      column-gutter: 0.25cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.cell(colspan: 9, inset: (top: 6pt, bottom: 4pt))[*Panel A.* Allocations by arrangement],
      table.hline(stroke: 0.4pt),
      [*Arrangement*], [corr$(alpha, nu)$], [$macron(mu)$], [$K^(M E)$], [$K^(P E)$], [$C^(M E)$], [$C^(P E_I)$], [$C^U$], [$C^(P E)$],
      [Baseline], [0.78], [1.184], [109.4], [234.2], [60.63], [64.12], [62.90], [79.99],
      [Shuffled ($times 10$)], [0.01], [1.174], [94.4], [160.1], [55.43], [57.11], [55.60], [65.08],
      [Reverse-sorted], [$-0.82$], [1.172], [88.3], [143.1], [53.36], [54.77], [53.36], [61.35],
      [Homogeneous $alpha$], [0.09], [1.173], [91.4], [140.1], [55.57], [56.92], [55.58], [62.71],
      table.hline(stroke: 0.7pt),
    )
    #v(0.1cm)
    #table(
      columns: (2.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
      column-gutter: 0.25cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.cell(colspan: 6, inset: (top: 6pt, bottom: 4pt))[*Panel B.* Welfare legs by arrangement, $lambda$],
      table.hline(stroke: 0.4pt),
      [], table.cell(colspan: 2, align: center)[_Lens A_], table.cell(colspan: 2, align: center)[_Lens B_], [],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*Arrangement*], [$lambda_"realloc"$], [$lambda_"scale"$], [$lambda_"disp"$], [$lambda_"level"$], [$lambda_"total"$],
      [Baseline], [5.76%], [24.75%], [3.75%], [27.18%], [31.94%],
      [Shuffled ($times 10$)], [3.04%], [13.95%], [0.32%], [17.04%], [17.41%],
      [Reverse-sorted], [2.63%], [12.01%], [$-0.01%$], [14.97%], [14.96%],
      [Homogeneous $alpha$], [2.44%], [10.17%], [0.02%], [12.83%], [12.85%],
      table.hline(stroke: 0.7pt),
    )
    #v(0.1cm)
    #table(
      columns: (2.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
      column-gutter: 0.25cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.cell(colspan: 6, inset: (top: 6pt, bottom: 4pt))[*Panel C.* Split of each baseline leg by @eq:sorting_split, $Delta$ (share of leg)],
      table.hline(stroke: 0.4pt),
      [], table.cell(colspan: 2, align: center)[_Lens A_], table.cell(colspan: 2, align: center)[_Lens B_], [],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*Term*], [$Delta_"realloc"$], [$Delta_"scale"$], [$Delta_"disp"$], [$Delta_"level"$], [$Delta_"total"$],
      [Common $alpha$], [0.0241 (42.9%)], [0.0968 (43.8%)], [0.0002 (0.4%)], [0.1207 (50.2%)], [0.1209 (43.6%)],
      [Heterogeneity], [0.0059 (10.5%)], [0.0338 (15.3%)], [0.0030 (8.3%)], [0.0366 (15.2%)], [0.0396 (14.3%)],
      [Sorting], [0.0261 (46.6%)], [0.0906 (41.0%)], [0.0336 (91.3%)], [0.0831 (34.6%)], [0.1167 (42.1%)],
      table.hline(stroke: 0.4pt),
      table.cell(inset: (top: 4pt, bottom: 4pt))[Baseline $Delta$],
      table.cell(inset: (top: 4pt, bottom: 4pt))[0.0560],
      table.cell(inset: (top: 4pt, bottom: 4pt))[0.2212],
      table.cell(inset: (top: 4pt, bottom: 4pt))[0.0368],
      table.cell(inset: (top: 4pt, bottom: 4pt))[0.2404],
      table.cell(inset: (top: 4pt, bottom: 4pt))[0.2772],
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ All arrangements are solved on the same draws of $nu_(j i)$, the same active set and anchor, and with $chi$ at its baseline value, and shuffled values are averaged over ten permutation draws. In Panel A, $macron(mu)$ is the market's cost-weighted markup, which the uniform-markup allocation preserves, the fixed-input planner holds $K = K^(M E)$, and under homogeneous $alpha$ the correlation comes from variation across sectors only. Panel B reports the legs of @eq:lens_a and @eq:lens_b, which multiply to $1 + lambda_"total"$ within each lens. In Panel C, the common-$alpha$ term uses the primitive homogeneous-$alpha$ variant, and standard errors of the heterogeneity and sorting terms across draws are at most $0.0001$.])
    #v(0.5em)
  ],
  caption: [*Welfare cost of markups by $alpha$-arrangement.*],
)<tab:sorting>
#v(0.3cm)

*Lens B Decomposition: Dispersion $times$ Aggregate Level.* The dispersion leg, by contrast, depends almost entirely on sorting. Shuffling $alpha$ lowers $lambda_"disp"$ from $3.75%$ to $0.32%$, and reverse sorting or homogeneous $alpha$ bring it to essentially zero ($-0.01%$ and $0.02%$, respectively), as can be seen in Panel B of @tab:sorting. Sorting then accounts for a substantial $91.3%$ share of $Delta_"disp"$, against $8.3%$ for heterogeneity and $0.4%$ for the common-$alpha$ term (Panel C). Therefore, this result suggests that almost all of the welfare cost of markup dispersion found in @sec:welfare_baseline comes from the pairing of scalability with capability. This could potentially be explained by the fact that, under positive sorting, the firms with the highest markups are also the most scalable, and whose output responds the most to a price wedge since their marginal cost is the least convex due to high $alpha$. Since the markup tail is untargeted, the absolute size of $lambda_"disp"$ should be taken with a grain of salt, whereas its collapse across arrangements rests on the same draws and is less exposed to it.

Compared to dispersion, the level leg is far less dependent on sorting, with sorting accounting for $34.6%$ of $Delta_"level"$ and the common-$alpha$ term for $50.2%$, the largest of the three. Interestingly, however, $lambda_"level"$ still falls by more than a third when $alpha$ is shuffled, from $27.18%$ to $17.04%$, although the aggregate wedge $macron(mu) - 1$ that the uniform-markup allocation preserves only falls from $0.184$ to $0.174$ (Panel A). A potential cause could stem from the same capital margin as in Lens A, since removing a given aggregate wedge frees more capital when the most scalable technologies sit in the most capable firms, but I have not isolated this channel. Therefore, the cost of a given markup level depends on the technologies that sit behind it, which is the common thread across the four legs that I take up in the synthesis.


*Full effect. * Ranked by the sorting share of their $Delta$, the four legs run from dispersion ($91%$) through reallocation ($47%$) and scale ($41%$) to level ($35%$), with sorting at $42%$ of $Delta_"total"$, as reported in Panel C of @tab:sorting. These findings make a strong case for the idea the cost of market power depends on which technologies carry the wedges, since at the same marginal distributions and with an aggregate wedge $macron(mu) - 1$ that differs by only $0.01$, the pairing of scalability with capability alone moves $lambda_"total"$ from $17.4%$ to $31.9%$. A potential critique is that the welfare impact is merely rising linearly in corr$(alpha, nu)$ from -1 to 1. To investigate this potential issue, I map out this nonlinearity by adding intermediate arrangements randomly shuffling $alpha$ among a fraction of active firms within each sector: $20%$, $40%$, $60%$, and $80%$ starting from the baseline, and $33%$ and $67%$ starting from reverse sorting. Each permutation preserves the sectoral distributions of scalability and capability, the active set, and the baseline parameters. I display the resulting relationship in @fig:sorting_convexity in the Appendix, making sure to re-solve all four allocations and calculate the welfare legs so that all points in the figure are solved economies plotted at their realized corr$(alpha, nu)$.

The points show that welfare costs rise slowly over negative correlations and much faster over positive correlations. The shuffled arrangement sits $52%$ of the way between the endpoints in correlation, but only $9%$ to $18%$ of the way in welfare cost. Thus, increasing correlation matters much more when it strengthens positive sorting than when it weakens negative sorting. This is the sense in which the relationship appears convex: the positive branch has a steeper slope, although it is approximately linear and the figure does not establish an exact threshold at zero. This result shows clearly that the impact of positive sorting between $alpha$ and $nu$ is the real force behind the results above, not merely an increasing correlation.


#v(0.5cm)
= Conclusion<sec:conclusion>
#v(0.5cm)


#colbreak()
#bibliography("literature.bib", title: "References", style: "harvard-cite-them-right")

#colbreak()

// Appendix
#counter(heading).update(0)
#set heading(numbering: "A.1.", supplement: [Appendix])


= Derivations
#v(0.8cm)

== Demand and Markup Derivations
#v(0.2cm)

Conditional on market-level expenditure, the CES block in @eq:market_ces implies the demand system in @eq:firm_demand and the market-share expression in @eq:share_def. Under Cournot competition, firm $i$ internalizes the effect of its own quantity on the market price index and hence on its residual demand elasticity. The perceived elasticity can be written as
#math.equation(block:true, $epsilon_(j i,t)^(-1) = frac(1-s_(j i,t), gamma) + frac(s_(j i,t), eta),$)<eq:perceived_elasticity>
so that
#math.equation(block:true, $mu_(j i,t) = frac(epsilon_(j i,t), epsilon_(j i,t)-1).$)<eq:markup_eps>
Substituting @eq:perceived_elasticity into @eq:markup_eps yields the inverse-markup expression @eq:emx_markup reported in the text. This inverse-markup system is the object that must be solved before aggregation and moment construction.

As a pooled diagnostic, cost-weighting firm revenue-cost wedges across all producing firms yields the aggregate cost-weighted markup
#math.equation(block:true, $mu_("cw")^("model") = integral_0^1 sum_(i in A_j) lambda^c_(j i) mu_(j i) d j,$)<eq:mu_cw_identity>
where $lambda^c_(j i)$ is firm $i$'s share of aggregate variable cost. This pooled cost-weighted markup is the model analogue of the SG&A-inclusive aggregate markup target $mu_("cw,sga")^("data")$, which is itself constructed as a cost-weighted mean of firm-level price-marginal-cost ratios. It should not be confused with the accounting ratio of aggregate revenue to aggregate variable cost: because @eq:total_cost and @eq:pricing imply $T C_(j i) = alpha_(j i) p_(j i) y_(j i) slash mu_(j i)$, that ratio is instead
#math.equation(block:true, numbering: none, $mu^alpha_("cw") = integral_0^1 sum_(i in A_j) lambda^c_(j i) frac(mu_(j i), alpha_(j i)) d j,$)<eq:mu_cw_alpha>
which exceeds $mu_("cw")^("model")$ whenever $alpha_(j i) < 1$, since it also loads on the variable profit that decreasing returns generate even at a unit markup. The two coincide only under constant returns; at the calibrated point they are $1.185$ and $1.402$. Only $mu_("cw")^("model")$ is targeted, and $mu^alpha_("cw")$ is reported as a diagnostic. The firm-level inverse-markup identity @eq:emx_markup implies that sector HHI summarizes how concentration moves sector inverse markups. The model-side `emx_slope` is therefore the slope from a static cross-sector regression of sector inverse cost-weighted markup on sector HHI.

== Cost and Technology Derivations
#v(0.2cm)

Given the inner value-added composite $v = k^(a) l^(1-a)$, the firm solves
#math.equation(block:true, numbering: none, $min_(k,l) R_t k + W_t l quad "s.t." quad v = k^(a) l^(1-a).$)
The first-order conditions imply the standard Cobb-Douglas input ratio
#math.equation(block:true, numbering: none, $frac(k_(j i,t), l_(j i,t)) = frac(a, 1-a) frac(W_t, R_t),$)<eq:input_ratio>
and the corresponding value-added unit cost index @eq:omega. Combining value added with materials gives the gross-output unit cost @eq:omega_gross. Substituting the production technology @eq:production into the expenditure function gives total variable cost @eq:total_cost, marginal cost @eq:marginal_cost in anchored form, and the conditional input demands @eq:input_demand.

== Planner Derivations<sec:planner_appendix>
#v(0.2cm)

Let the planner's Lagrangian attach multipliers $lambda_t^L$ and $lambda_t^K$ to the aggregate labor and capital constraints and let $psi_(j i,t)$ denote the marginal contribution of one more unit of firm output to the gross composite bundle. The planner's firm-level FOCs are the conditions reported in @eq:planner_labor_foc, @eq:planner_capital_foc, and @eq:planner_materials_foc. Together with @eq:production, they imply
#math.equation(block:true, numbering: none, $U_C(C_t,L_t) psi_(j i,t) alpha_(j i) phi.alt_v (1-a) frac(y_(j i,t), l_(j i,t)) = lambda_t^L,$)
#math.equation(block:true, numbering: none, $U_C(C_t,L_t) psi_(j i,t) alpha_(j i) phi.alt_v a frac(y_(j i,t), k_(j i,t)) = lambda_t^K,$)
#math.equation(block:true, numbering: none, $psi_(j i,t) (1-phi.alt_v) alpha_(j i) frac(y_(j i,t), m_(j i,t)) = 1.$)
The planner keeps the same exogenous sector firm set $A_j$ as the decentralized allocation. Hence the planner derivation needs no additional firm-selection condition: all differences between the market and planner allocations come from markup pricing and the induced allocation of labor, capital, materials, and output.

== Equilibrium Accounting
#v(0.2cm)

The decentralized equilibrium is summarized by the household block, the sector-level demand system, firm markup pricing, the exogenous Poisson firm-count draw, the resource constraint, and the aggregate labor and capital constraints. This appendix section serves as the reference map for the later steady-state solution and for the computational routines that solve the sector-by-sector fixed point.

For the decentralized steady state, the unknowns can be grouped as:
#math.equation(block:true, numbering: none, ${C, Q, M, w, R, {Y_j, P_j, A_j, {y_(j i), p_(j i), s_(j i), mu_(j i), d_(j i), m_(j i)}_(i in A_j)}_(j in [0,1])}.$)
For the planner steady state, the corresponding unknowns replace markup pricing with planner shadow prices while keeping the same exogenous firm set $A_j$:
#math.equation(block:true, numbering: none, ${C^*, Q^*, M^*, w^("plan"), R^("plan"), {Y_j^*, A_j^*, {y_(j i)^*, k_(j i)^*, l_(j i)^*, m_(j i)^*, psi_(j i)^*}_(i in A_j^*)}_(j in [0,1])}.$)
These lists are not yet a numerical algorithm; they are the bookkeeping device for the later steady-state solve.

= Additional discussion

#v(0.5cm)
== Choice of anchoring statistic<anchor>
#v(0.2cm)

The anchoring statistic is the median rather than the sales-weighted mean of active-firm output, and this choice is not innocuous. By @eq:marginal_cost, the returns-to-scale term $(y/hat(y))^(1/alpha_(j i)-1)$ is exactly neutral at $y = hat(y)$, so $alpha_(j i)$ only distorts marginal cost for firms whose output falls away from the anchor: wherever the anchor sits, that is the population of firms $alpha_(j i)$ is not penalizing, and every other firm is priced relative to it.

Active-firm output is heavily right-skewed, so the sales-weighted mean is pulled deep into the tail of that distribution: in the baseline draw it exceeds the output of roughly 99.5% of active firms. Anchoring there would make $alpha_(j i)$ act, for nearly the entire firm population, as a mechanical penalty for being smaller than a handful of superstar firms, mixing genuine curvature in the production technology with the separate, already well-documented phenomenon of firm-size concentration.

The median sidesteps this by construction: it splits the active-firm population evenly on either side of $hat(y)$ regardless of how fat the right tail is, so $alpha_(j i)$ is disciplined by curvature around a typical operating scale rather than by distance from the largest incumbents. The choice of median is also robuts, with re-anchoring $hat(y)$ to the baseline market's median active output, held fixed across arrangements, moves everything less than 1%.

#v(0.5cm)
== Full parameterization<sec:full_param>
#v(0.2cm)

#figure(
  block(width: 96%)[
    #set text(size: 8pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (1.15fr, 3.35fr, 3.50fr),
      column-gutter: (0.25cm, 0.25cm),
      row-gutter: 0.12cm,
      align: left,
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.header[*Parameter*][*Description*][*Value / Status*],
      table.hline(stroke: 0.4pt),

      table.cell(colspan: 3, align: left)[_A. Assigned parameters_],
      [$beta$], [discount factor], [$0.96$],
      [$delta_K$], [capital depreciation rate], [$0.06$],
      [$phi$], [Frisch elasticity of labor supply], [$1.00$],
      [$a$], [common capital share inside value added], [$0.3946$; Empirical VA-weighted KLEMS $a_"bar"$],
      [$hat(y)$], [common anchor output (units normalization)], [$equiv 1$],
      table.hline(stroke: 0.7pt),

      table.cell(colspan: 3, align: left)[_B. External empirical inputs_],
      [$F^alpha$], [pooled scalability distribution], [firm-level gross-output returns to scale],
      [$alpha_(j i)$], [gross-output returns to scale], [external input],
      [$phi.alt_v$], [value-added weight in gross output], [$0.348$; Hybrid GNR-KLEMS],
      [$omega_j$], [sector exposure weights], [normalized to uniform in the current baseline],
      table.hline(stroke: 0.7pt),

      table.cell(colspan: 3, align: left)[_C. Parameters calibrated inside model_],
      [$xi$], [Pareto tail index of capability $nu$], [16.80; concentration tail],
      [$N$], [Poisson mean firms per sector], [57; sector firm count],
      [$gamma$], [within-market elasticity (common)], [7.25; markup level through $mu_("cw,sga")^("data")$],
      [$eta$], [across-market elasticity (common)], [1.544; Autor et al. slope through sector $1 slash mu_i ~ "HHI"$],
      [$macron(rho)$], [rank-copula strength of $alpha$--$nu$], [0.897; targets raw corr($alpha$, log sales) $= +0.596$, maps to corr($nu, alpha) approx 0.785$ through the nonlinear Pareto transform],
      [$underline(nu)$], [capability scale / Pareto lower bound], [195.48; solved by $tilde(y) = hat(y)$; not moment-targeted],
      [$nu_(j i)$], [capability primitive (inverse MC at anchor $hat(y)$)], [see @eq:nu_ladder and @eq:nu_pareto],
      table.hline(stroke: 0.7pt),

      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ The calibrated vector is $(xi, N, gamma, eta, macron(rho))$: $(xi, N)$ discipline concentration, $gamma$ pins the markup level, $eta$ is disciplined by the EMX/Autor slope, and $macron(rho)$ targets the raw $alpha$--sales correlation. The capability scale $underline(nu)$ is solved by the baseline anchoring condition $tilde(y) = hat(y) equiv 1$, not calibrated. External empirical inputs include $F^alpha$, $alpha_(j i)$, $phi.alt_v$, $a$, and $omega_j$.])
    #v(0.5em)
  ],
  caption: [*Parameterization used in the pooled sector-market steady-state implementation.*],
)<tab:param_ss>

#colbreak()

= Computational Details

#v(0.5cm)
== Solving Steady State<algo_ss>
#v(0.2cm)

This section will report the numerical algorithm used to draw sector firm counts, solve the market-level fixed point, aggregate outcomes across simulated sectors, and iterate on $(xi, N, gamma, eta, macron(rho))$ until the model matches its target moments.

#v(0.5cm)
== Solving Sector Firm Counts<sec:sector_count_algo>
#v(0.2cm)

For each simulated sector $j$, the code draws
#math.equation(block:true, numbering: none, $n_j = max(1, "Poisson"(N)),$)
and sets $A_j = {1, dots, n_j}$. The first $n_j$ slots in the sector's independently drawn firm pool are then passed to the Cournot solver. Since slots are exchangeable, this is equivalent to drawing an unordered $n_j$-firm sample. The only guard is numerical: if the Poisson draw exceeds the available slot count $H$, the draw is capped and the run is flagged so $H$ can be raised.

Given $A_j$, the solver computes the fixed point of @eq:firm_demand through @eq:profits once for that sector. The planner solve reuses the same $A_j$ and changes only the pricing/allocation rule.


#v(0.5cm)
= Aditional Figures <aditional_figs>
#v(0.2cm)



#figure(
  block(width: 80%)[
    #image("figures/fig2_alpha_raw_vs_winsorized_vs_clipped.pdf", width: 100%)
    #align(left, text(size: 7.5pt)[_Note._ Raw pooled hybrid GNR-KLEMS returns-to-scale estimates, the winsorized and discretized distribution, and the final clipped support passed to the solver.])
    #v(0.5em)
  ],
  caption: [*Construction of the scalability support $F^alpha$.*],
)<fig:alpha_support>

#v(0.5cm)

#figure(
  block(width: 80%)[
    #image("figures/fig_sorting_convexity_curve.pdf", width: 100%)
    #align(left, text(size: 7.5pt)[_Note._ Each leg's $Delta$ is normalized to zero under reverse sorting and one under the baseline, so the shuffled point, reported in the legend, is the fraction of the reverse-to-baseline movement reached at corr$(alpha, nu) approx 0$. The dashed line is the linear benchmark between the two endpoints, which places the shuffled point at $52%$ in every leg, and the segments between the three arrangements are interpolations.])
    #v(0.5em)
  ],
  caption: [*Welfare cost of each leg against corr$(alpha, nu)$ across $alpha$-arrangements.*],
)<fig:sorting_convexity>
