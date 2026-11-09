#import "@preview/ssrn-scribe:0.10.1": paper

// Citations and cross-references (equations, figures, tables, sections) in blue.
// The template colours links with its accent, so the colour is set on links
// inside refs and cites rather than on the outer text.
// LaTeX hyperref's blue (pure RGB blue), not Typst's softer built-in `blue`.
#let ref-blue = rgb(0, 0, 255)
#show ref: it => { show link: set text(fill: ref-blue); set text(fill: ref-blue); it }
// As with natbib + hyperref, the parentheses around the year stay in the text colour.
#show cite: it => {
  show link: set text(fill: ref-blue)
  set text(fill: ref-blue)
  show regex("[()]"): set text(fill: rgb("#111111"))
  it
}

// Text that still needs rewriting for the Orbis Spain results.
// highlight() skips math glyphs, so math inside a flag gets a yellow box.
#let flag(body) = {
  show math.equation.where(block: false): it => box(fill: yellow, it)
  highlight(fill: yellow, body)
}

#show: paper.with(
  meta: (
    title: [Market Power and the Scalability of Firms],
    authors: (
      (
        name: "Enrico Truzzi",
        affiliation: "Universitat Pompeu Fabra",
        email: "enrico.truzzi@upf.edu",
      ),
    ),
    abstract: [#h(1.5em)Standard oligopoly models let firms differ in productivity but assign every firm the same returns to scale. In this paper, I introduce firm-level heterogeneity in returns to scale, disciplined by gross-output production function estimates for Spanish firms in Orbis, into an Atkeson-Burstein-style Cournot oligopoly and study how the sorting between productivity and scalability shapes the welfare cost of market power. The sorting is calibrated to how returns to scale vary with firm size in the data. In the calibrated economy, removing markups raises consumption by $32.3%$, more than twice the gain in an economy with common returns to scale, as productivity and scalability act as complements. The bulk of the gap comes from the scale of production rather than from the misallocation of a given stock of inputs, since wedges hold back output most at the firms with the highest returns to scale. Holding both marginal distributions fixed, the gain is S-shaped in the correlation between the two margins, rising from $9.0%$ under perfect negative pairing to $50.3%$ under random reassignment. The welfare cost of market power therefore depends on which firms carry the markups.],
    date: datetime.today().display("[month repr:long] [day padding:none], [year]"),
    keywords: [Markups, Returns to scale, Scalability, Sorting, Market power, Misallocation, Oligopoly],
  ),
  theme: (
    font: "New Computer Modern",
    heading-font: "New Computer Modern",
  ),
  layout: (
    maketitle: false,
    // The template leaves the abstract box (86% wide) left-aligned, which
    // looks off-centre; full text width keeps it flush with the margins.
    cover-text-width: 100%,
    cover-spacing: 28pt,
    frontmatter-gap: 12pt,
    density: "balanced",
    // Slightly tighter than the template default of 2.54cm on every side.
    margin: (left: 2.35cm, right: 2.35cm, top: 1.9cm, bottom: 2.0cm),
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
// Main-body figures and tables float to the top or bottom of the page where
// they fit best (LaTeX-style). Reset to fixed placement at the appendix below.
#set figure(placement: auto)

// Equation references render as the bare number, e.g. "Equation (3)" is written
// in the text as "Equation @eq:...".
#show ref: it => {
  let eq = math.equation
  let el = it.element
  if el != none and el.func() == eq {
    show link: set text(fill: ref-blue)
    link(el.location(), numbering(
      el.numbering,
      ..counter(eq).at(el.location())
    ))
  } else {
    it
  }
}



#v(0.5cm)
= Introduction
#v(0.2cm)

Firm heterogeneity plays a key role in the current literature addressing the rise in market power, in particular in explaining the documented rise in markups over recent decades. Seminal models of firm heterogeneity, such as #cite(form:"prose",<melitz2003impact>), #cite(form:"prose",<atkeson2008pricing>) and #cite(form:"prose",<hsieh2009misallocation>), make firm size depend on productivity alone, and in #cite(form:"prose",<atkeson2008pricing>) more productive firms also capture larger market shares and therefore charge higher markups. Heterogeneous markups then distort both the allocation of demand across firms and the overall level of production #cite(<baqaee2020productivity>) #cite(<edmond2023costly>) #cite(<de2020rise>). Such models normally assume homogeneous _returns to scale_ (_RTS_) across firms, letting heterogeneity in _total factor productivity_ (_TFP_) explain most of the origin of firm markups. Recent works such as #cite(form:"prose",<lashkari2024information>), #cite(form:"prose",<argente2025scalable>), #cite(form:"prose",<demirer2025production>) and #cite(form:"prose",<hubmer2025scalable>), however, have shown that firms differ in how easily they scale, and that this heterogeneity plays an important role in explaining key facts of firm dynamics previously associated with TFP differences. In particular, #cite(form:"prose",<hubmer2025scalable>) find in their quantitative implementation that heterogeneity in RTS more than doubles the efficiency losses from financial frictions, compared with an economy where firms differ in TFP alone.

In this paper, I introduce firm-level heterogeneity in returns-to-scale parameters to a standard #cite(<atkeson2008pricing>, form: "prose")-style Cournot oligopoly framework, and explore how important this joint heterogeneity of productivity and scalability and its alignment are for the welfare impact of market power. With heterogeneous RTS, productivity and scalability jointly determine equilibrium size, and, under oligopoly, size determines markups. I argue that the sorting between fundamental productivity, which determines a firm's inverse marginal cost at a common reference output, and scalability, and not only their absolute value, plays a major role in shaping the aggregate welfare costs of market power. The mechanism is that a given markup wedge is more distortionary when it is attached to a firm that can absorb large quantities with only a limited increase in cost.

To achieve this, I introduce a quantitative framework designed to incorporate both variation in productivity and RTS within a pre-established and well-known oligopoly environment. The setting is an Atkeson-Burstein-style sector oligopoly model with heterogeneous market exposure and finite-firm competition inside product markets. Producers are draws from a common continuous joint distribution $F(alpha, nu)$ where the RTS margin $alpha$ is externally disciplined, while the productivity margin $nu$ has a Pareto marginal with tail index $xi$ following #cite(<edmond2023costly>, form: "prose"). Returns to scale are disciplined by firm-level gross-output production function estimates based on the approach of #cite(<gandhi2020identification>, form: "prose") (henceforth GNR) in the spirit of #cite(<hubmer2025scalable>, form: "prose"). Meanwhile, firm productivity is summarized by its inverse marginal cost at a common anchor output, calibrated inside the model. Productivity and scalability are linked by a rank copula whose strength $macron(rho)$ is also calibrated, and the number of firms in each sector is drawn from a Poisson distribution with mean $N$. The within-market elasticity $gamma$ mainly disciplines the aggregate markup, which I measure following the production-function approach of #cite(form: "prose",<de2020rise>), as the product of the estimated variable-input elasticity and the firm's ratio of sales to materials and staff costs, aggregated across firms with cost weights that include the user cost of capital, yielding a target of $1.1815$. I use cost rather than sales weights since, as #cite(<edmond2023costly>, form: "prose") show, the sales-weighted markup equals the cost-weighted one plus a term that captures the variance of markups, and would therefore overstate the aggregate wedge.

I use firm-level data for Spanish firms in Orbis from 2015 to 2019, with monetary values expressed in 2019 euros, which leaves me with 1,391,061 firm-years for the estimation of returns to scale. For each NAICS2 cell, I estimate a gross-output production function with the GNR method, whose materials, labor and capital output elasticities vary across firm-years with their input mix, and sum the three into a firm-year RTS estimate, whose mean over the window forms the scalability distribution $F^alpha$#footnote[See @sec:data for more information. Since cost minimization requires returns to scale not to exceed the markup, scalability draws come from $F^alpha$ truncated at $0.99 gamma slash (gamma - 1)$, just below the CES floor.]. I do not target concentration measures, since measured concentration in Orbis depends on the representativeness of the firm sample #cite(<kalemli2024construct>) and on the market definition, and instead discipline the upper tail of firm size through two tail indices of firm sales.

In line with #cite(form: "prose",<hubmer2025scalable>), I find that returns to scale in the Orbis data are U-shaped in firm size, both within NAICS2 industry-years and in the pooled sample, with the top 5% of firms by sales reaching an $alpha$ about $0.074$ above the industry mean. As in their estimates, the steep rise at the top is driven by the output elasticity of intermediate inputs. Differently from the authors, however, the left arm of the U is more pronounced, since the labor elasticity falls with size.

The calibrated parameters are pinned down jointly to match the aggregate markup, the two tail indices of firm sales and the gap in returns to scale between the largest firms and the bottom half of the sales distribution, which identifies the sorting parameter. As in #cite(<hubmer2025scalable>, form: "prose"), I calibrate the dependence between productivity and scalability internally, since their joint distribution cannot be read off the production-function estimates, and like the authors, I find a negative correlation between the two, such that some highly scalable firms remain small while others reach the top of the sales distribution, in line with the U-shaped profile above.

Removing every markup raises steady-state consumption by $lambda = 32.3%$ in the calibrated economy#footnote[This value comes from the sorting experiments of @sec:sorting, which are solved on $M = 10000$ markets. The comparison across economies in @tab:welfare_hetero uses $M = 5000$, where the calibrated gain is $32.04%$, and the percentage-point contributions and shares reported for that comparison and its decomposition in @fig:recal_channels refer to this value.], which I refer to as the welfare gap between the market and the planner allocation. To understand which dimension of heterogeneity drives this gap, I compare this baseline $(alpha, nu)$-economy with three counterfactual economies, each recalibrated to the targets it can reach, such that the comparison holds the observed moments fixed rather than the parameters. The $nu$-economy holds RTS common at the harmonic mean of $F^alpha$, as in the standard approach to measuring the cost of markups #cite(<edmond2023costly>), and matches the same markup level and nearly matches the same sales tails as the baseline. However, its welfare gap is only $14.3%$, less than half of the calibrated one. The $alpha$-economy and the homogeneous economy, which both hold productivity common, reach $13.0%$ and $10.8%$, such that each margin on its own adds little to the cost of a common markup, with productivity dispersion contributing $3.48$ percentage points and scalability dispersion $2.19$. I find that the joint interaction of the two, by contrast, accounts for $15.6$ percentage points, or $48.7%$ of the calibrated gap. That is, productivity and scalability act as complements, and almost half of the welfare cost of markups arises only when firms differ along both dimensions at once.

To understand where this cost comes from, I split the welfare gap in two separate ways using intermediate planner allocations. The first separates the dispersion of markups across firms from their common level #cite(<baqaee2020productivity>) #cite(<edmond2023costly>), and attributes $43%$ of the gap to dispersion. The second asks whether the planner gains by reallocating the inputs that the market already uses or by using more of them, in the spirit of #cite(<hsieh2009misallocation>, form: "prose") and #cite(<restuccia2008policy>, form: "prose"), and finds that reallocation alone would raise consumption by only $5.5%$, while capital accounts for $63%$ of the gap. Therefore, markups in this economy lower welfare mostly by holding back the expansion of scalable firms, which corroborates the idea that the two margins act as complements. Productivity generates the fat tail of sales, and with it the large shares and high Cournot markups, while scalability determines how much output a given markup withholds, since the elasticity of a firm's output to its own markup rises with its returns to scale #cite(<ruzic2023returns>), going from $gamma = 7.88$ under constant returns to about $28$ at $alpha = 1.1$.

To measure how much the pairing matters for this complementarity, I hold the calibrated parameters and both marginal distributions fixed and vary only the strength of the rank copula, together with within-sector permutations that implement perfect negative and positive pairing and random reassignment. I find that the pairing moves the welfare gap by far more than any dimension of heterogeneity, with $lambda$ rising from $9.0%$ under perfect negative pairing to $50.3%$ under random reassignment and $152.3%$ under perfect positive pairing. The response is S-shaped in the correlation between scalability and productivity and steepest around the calibrated economy, such that $lambda$ doubles as the correlation moves from $-0.35$ to the calibrated $-0.14$, which makes the RTS gap that identifies sorting a first-order determinant of the cost of markups. Concluding that positive sorting lowers welfare would, however, be incorrect, since welfare rises with sorting in both the market and the planner allocation, with the gap widening because the planner rises faster.

Using the same two welfare gap decompositions as before, I find that sorting raises the cost of markups through two channels. The first is market power, since aligning scalability with productivity lets the most productive firms expand further, raising both the aggregate markup and the spread of markups. The second is the reallocation of inputs towards scalable firms, whose output responds most to a given markup. Consequently, the dispersion component grows from zero under strongly negative sorting to about half of the gap at the top, and misallocation to about a quarter, while capital remains the largest component throughout.
#v(0.3cm)
*Related Literature.* This paper contributes to three strands of literature. The first quantifies the aggregate cost of markups when firms are heterogeneous, where #cite(<edmond2015competition>, form: "prose"), #cite(<peters2020heterogeneous>, form:"prose"), #cite(<edmond2023costly>, form: "prose") and #cite(<deloecker2021quantifying>, form: "prose") measure it with a common returns to scale across firms. I add heterogeneity in returns to scale disciplined by production-function estimates, and show that the same markups are far more costly when they sit on scalable firms. The second strand documents that firms differ in their returns to scale. #cite(<hubmer2025scalable>, form: "prose") and #cite(<demirer2025production>, form: "prose") estimate production functions whose output elasticities, and thus returns to scale, vary across firms, with #cite(<hubmer2025scalable>, form: "prose") showing that returns to scale rise with size at the top of the distribution, while #cite(<kariel2024scale>, form: "prose") and #cite(<baqaee2024darwinian>, form: "prose") study the aggregate implications of scale economies. A related literature traces scalability to information technology #cite(<lashkari2024information>), scalable expertise #cite(<argente2025scalable>) and intangible inputs #cite(<deridder2024market>). I contribute to this strand by embedding returns to scale estimated on Spanish firms in an oligopoly with endogenous markups, which shows that returns to scale determine how much output a given markup withholds, and therefore the welfare cost of market power. Finally, #cite(<restuccia2008policy>, form: "prose"), #cite(<hsieh2009misallocation>, form: "prose") and #cite(<baqaee2020productivity>, form: "prose") show that wedges matter for aggregate outcomes through their covariance with producer characteristics, while #cite(<david2019sources>, form: "prose"), #cite(<ruzic2023returns>, form: "prose") and #cite(<zhang2022misallocation>, form: "prose") show that heterogeneity in technologies and markups changes how misallocation should be measured. In this paper, the sorting between productivity and scalability endogenously places the largest markups on the firms that can expand most easily, which accounts for a substantial share of the welfare gap.



#v(0.5cm)
= Theory
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
Gross composite output is split between household consumption, investment and intermediate use, such that
#math.equation(block:true, $C_t = Q_t - M_t - I_t, quad M_t = integral_0^1 sum_(i in A_(j,t)) m_(j i,t) d j,$)<eq:resource>
which in a steady state, where $I = delta_K K$, reads $C = Q - M - delta_K K$.
The exposure weight $omega_j$ summarizes how much aggregate expenditure is exposed to product market $j$ and is normalized so that $integral_0^1 omega_j d j = 1$. In the current baseline the simulated sectors are pooled with normalized exposure weights so they all have the same weight. Materials are purchases of the same composite at price $P_t$. The expenditure allocated to market $j$ is $X_(j,t) equiv P_(j,t) Y_(j,t)$, and the corresponding demand schedule loads on total composite use:
#math.equation(block:true, numbering: none, $Y_(j,t) = omega_j (P_(j,t)/P_t)^(-eta) Q_t.$)<eq:sector_demand>

#v(0.5cm)
== Demand and Market Shares
#v(0.2cm)

*Nested demand structure.* There is a continuum of finite sectors indexed by $j in [0,1]$. Each sector is one product market. Let $X_(j,t)$ denote composite expenditure assigned to sector $j$. Within sector $j$ a finite number $n_(j,t)$ of firms compete. Firm $i in {1, ..., n_(j,t)}$ supplies quantity $y_(j i,t)$ and charges price $p_(j i,t)$. Conditional on sector expenditure, the within-sector composite good is CES. Throughout the baseline I write the aggregator and its price index with an explicit $1 slash n_(j,t)$ weight that removes mechanical love of variety#footnote[Under symmetry, $n$ firms each producing $y$ would deliver an unweighted composite $n^(gamma/(gamma-1)) y$, which rises with the number of firms at fixed per-firm output. The weight removes this term and yields $Y_(j,t) = y$, so that only reallocation and heterogeneity gains remain. Conditional on sector expenditure $X_(j,t)$, it rescales $Y_(j,t)$ and $P_(j,t)$ by reciprocal powers of $n_(j,t)$ and leaves every firm-level quantity, price, share, markup and profit unchanged. Only the within-sector nest carries this normalization, since the exposure weights $omega_j$ are fixed and @eq:agg_ces has no variety margin.], where $n_(j,t) equiv |A_(j,t)|$ is the realized number of allocated producing firms (the set $A_(j,t)$ is defined in @sec:market_structure below):
#math.equation(block:true, $Y_(j,t) = [sum_(i in A_(j,t)) frac(1, n_(j,t)) y_(j i,t)^((gamma-1)/gamma)]^(gamma/(gamma-1)), quad gamma > eta,$)<eq:market_ces>
with price index
#math.equation(block:true, numbering: none, $P_(j,t) = [(frac(1, n_(j,t)))^gamma sum_(i in A_(j,t)) p_(j i,t)^(1-gamma)]^(1/(1-gamma)) = n_(j,t)^(gamma/(gamma-1)) [sum_(i in A_(j,t)) p_(j i,t)^(1-gamma)]^(1/(1-gamma)).$)<eq:market_price>
The within-market elasticity of substitution $gamma$ is common across markets: it governs how aggressively firms inside a market substitute against one another. The across-market elasticity $eta$ is a single global object governing substitution across product markets.

#v(0.3cm)

*Shares and markups.* The conditional demand for firm $i$ in market $j$ is therefore
#math.equation(block:true, $y_(j i,t) = (frac(1, n_(j,t)))^gamma (p_(j i,t)/P_(j,t))^(-gamma) Y_(j,t),$)<eq:firm_demand>
and its revenue share inside the market is
#math.equation(block:true, $s_(j i,t) equiv frac(p_(j i,t) y_(j i,t), sum_h p_(j h,t) y_(j h,t)) = frac(p_(j i,t)^(1-gamma), sum_h p_(j h,t)^(1-gamma)).$)<eq:share_def>

Finite-firm competition implies that firms internalize the effect of their own quantity choices on the market price index. In the nested-CES Cournot environment studied here and by #cite(<edmond2023costly>, form: "prose") and by #cite(<atkeson2008pricing>, form: "prose"), this generates an inverse-markup rule of the form
#math.equation(block:true, $frac(1,mu_(j i,t)) = 1 - frac(1,gamma) - (frac(1,eta) - frac(1,gamma)) s_(j i,t).$)<eq:emx_markup>
Equation @eq:emx_markup is the key pricing object in the model. Because $gamma > eta$, the coefficient on $s_(j i,t)$ is negative, implying that firms with larger market shares face less elastic residual demand and charge higher markups. With a common $gamma$, the within-sector markup floor as $s_(j i,t) -> 0$ is the same across sectors, $mu_min = gamma/(gamma-1)$; all heterogeneity in realized markups comes from endogenous market shares, which in turn reflect the distribution of firm draws within each sector and the exogenous sector firm count. The detailed Cournot derivation is relegated to the appendix.

#v(0.5cm)
== Firms and Technology
#v(0.2cm)

*Heterogeneity.* Firms differ along two dimensions. First, they differ in scalability, indexed by $alpha_(j i)$, which governs gross-output returns to scale and how fast marginal cost rises with output. Second, they differ in fundamental productivity, indexed by $nu_(j i)$, the inverse-marginal-cost shifter that fixes each firm's marginal cost at a common anchor output $hat(y)$: a higher $nu_(j i)$ means a lower marginal cost at the anchor. Fundamental productivity $nu_(j i)$ is distinct from the production-function TFP coefficient $z_(j i)$ introduced below. The empirical strategy treats $alpha_(j i)$ as externally disciplined and treats $nu_(j i)$ as the productivity primitive. Henceforth, productivity refers to $nu_(j i)$. The dependence between $alpha$ and $nu$ is introduced only through ranks. Let $tilde(alpha)_(j i)$ be the standard-normal score of firm $i$'s scalability rank in sector $j$, and let $epsilon_(j i) ~ N(0,1)$. The productivity ladder is
#math.equation(block:true, $ell_(j i) = macron(rho) tilde(alpha)_(j i) + sqrt(1 - macron(rho)^2) epsilon_(j i),$)<eq:nu_ladder>
and the ladder percentile is mapped into a Pareto productivity draw,
#math.equation(block:true, $u_(j i) = Phi(ell_(j i)), quad nu_(j i) = underline(nu) (1 - u_(j i))^(-1/xi), quad xi > 1.$)<eq:nu_pareto>
The Pareto tail index $xi$ is calibrated to the tail indices of firm sales, the lower bound $underline(nu)$ is a units-fixing scale solved by the baseline anchoring condition (not moment-targeted), and the alignment of $nu$ with $alpha$ is set through the calibrated rank-copula parameter $macron(rho)$, which fixes the strength of the $alpha$--$nu$ rank copula. It is identified by the gap in mean returns to scale between the top 5% and the bottom 50% of firms by sales, $0.046$, a directly observed cross-sectional moment requiring no productivity proxy and no data-side level anchor (see @sec:calibration).

#v(0.3cm)

*Technology and Anchoring.* Let $phi.alt_v$ denote the value-added weight. Production in firm $i$ of market $j$ is gross output, combining the value-added composite $v_(j i,t) equiv k_(j i,t)^(a) l_(j i,t)^(1-a)$ with materials $m_(j i,t)$ in a constant-returns Cobb-Douglas input bundle $x_(j i,t) equiv v_(j i,t)^(phi.alt_v) m_(j i,t)^(1-phi.alt_v)$:
#math.equation(block:true, $y_(j i,t) = z_(j i) x_(j i,t)^(alpha_(j i)), quad x_(j i,t) = v_(j i,t)^(phi.alt_v) m_(j i,t)^(1-phi.alt_v), quad v_(j i,t) = k_(j i,t)^(a) l_(j i,t)^(1-a), quad phi.alt_v in (0,1).$)<eq:production>
The parameter $a$ is the common capital share inside value added. The value-added weight $phi.alt_v$ and materials weight $1-phi.alt_v$ give gross-output elasticities $(partial ln y)/(partial ln m) = (1-phi.alt_v)alpha_(j i)$, $(partial ln y)/(partial ln k) = phi.alt_v a alpha_(j i)$, and $(partial ln y)/(partial ln l) = phi.alt_v (1-a) alpha_(j i)$, summing to $alpha_(j i)$. The anchored parameterization takes fundamental productivity $nu_(j i)$ as the primitive and treats the TFP coefficient $z_(j i)$ as a derived object, with the two being linked by the relabeling identity
#math.equation(block:true, $z_(j i) = (nu_(j i) \/ alpha_(j i))^(alpha_(j i)) hat(y)^(1 - alpha_(j i)),$)<eq:relabel>
so @eq:production, the input bundle, and the conditional input demands are algebraically unchanged. Only the distributional assumption moves: the common Pareto is now asserted on the dimensionally coherent cost shifter $nu_(j i)$ (marginal cost at the common anchor $hat(y)$) rather than on $z_(j i)$, whose units differ across firms when $alpha_(j i)$ differs.

#v(0.3cm)

*Cost minimization.* Cost minimization in the inner value-added problem implies the common unit cost index
#math.equation(block:true, $Omega(W_t,R_t) = (R_t/a)^(a) (W_t/(1-a))^(1-a).$)<eq:omega>
With materials purchased at price $P_t$, the gross-output unit cost is
#math.equation(block:true, $Omega^g_t equiv Omega^g (W_t,R_t,P_t) = (Omega(W_t,R_t)/phi.alt_v)^(phi.alt_v) (P_t/(1-phi.alt_v))^(1-phi.alt_v).$)<eq:omega_gross>
Given the production technology in @eq:production, expressed through the productivity primitive $nu_(j i)$ at the anchor $hat(y)$, variable cost can be written in anchored form as
#math.equation(block:true, $T C_(j i,t)(y) = alpha_(j i) frac(Omega^g_t, nu_(j i)) hat(y) (y/hat(y))^(1/alpha_(j i)),$)<eq:total_cost>
which delivers marginal cost
#math.equation(block:true, $M C_(j i,t)(y) = frac(Omega^g_t, nu_(j i)) (y/hat(y))^(1/alpha_(j i)-1).$)<eq:marginal_cost>
At the anchor $y = hat(y)$, marginal cost is $M C_(j i,t)(hat(y)) = Omega^g_t \/ nu_(j i)$ regardless of $alpha_(j i)$: productivity alone ranks firms at the anchor, and $alpha_(j i)$ only tilts how marginal cost moves as output deviates from it, so the level channel and the curvature channel are separated by construction.#footnote[The anchored form is an algebraic identity, since writing the original total cost $Omega^g_t (y/z_(j i))^(1/alpha_(j i))$ as $Omega^g_t (hat(y)/z_(j i))^(1/alpha_(j i)) (y/hat(y))^(1/alpha_(j i))$ and substituting @eq:relabel gives @eq:total_cost.]
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

*Market Structure.* Each sector $j$ receives an exogenous number of producing firms,
#math.equation(block:true, $n_(j,t) = max(1, "Poisson"(N)),$)<eq:poisson_count>
where $N$ is a calibrated primitive and is interpreted as the mean number of firms per sector#footnote[The numerical implementation reserves $H$ firm slots per market, with $H$ large enough that the Poisson count rarely reaches it.]. Conditional on this draw, the set $A_(j,t) = {1, dots, n_(j,t)}$ contains the firms priced in the sector Cournot problem. All allocated firms produce in the static cross section. There is no per-period fixed production cost and no profitability-based removal of firms.

#v(0.3cm)

*Equilibrium.* Given the aggregate state $K_t$, the exogenous sector counts, and the distributions of $(alpha_(j i),nu_(j i))$, a decentralized equilibrium is a sequence of household allocations, firm allocations, prices, market shares, and sector firm sets such that (i) households optimize given prices and firm payouts, (ii) firms satisfy demand @eq:firm_demand and markup pricing @eq:pricing, (iii) each sector firm set $A_(j,t)$ is the exogenous Poisson allocation in @eq:poisson_count, (iv) composite feasibility holds as in @eq:resource, with materials clearing through $M_t = integral_0^1 sum_(i in A_(j,t)) m_(j i,t) d j$ and the aggregate price level normalized by $P_t equiv 1$, and (v) labor and capital services clear,
#math.equation(block:true, $L_t = integral_0^1 sum_(i in A_(j,t)) l_(j i,t) d j,$)<eq:labor_market>
#math.equation(block:true, $K_t = integral_0^1 sum_(i in A_(j,t)) k_(j i,t) d j.$)<eq:capital_market>

Aggregate accounting closes because gross sales equal $P_t Q_t$, variable cost equals factor payments plus materials purchases, and profits are the residual. Hence value added satisfies $P_t (Q_t-M_t) = P_t (C_t + I_t) = W_t L_t + R_t K_t + Pi_t$, since materials net out as intra-economy purchases and @eq:resource assigns the rest of the composite to consumption and investment.

#v(0.5cm)
== Planner's Allocation and Welfare<sec:planner_welfare>
#v(0.2cm)

*Planner Problem.*  The planner faces the same technology, the same exogenous sector firm sets, and the same distributions of scalability and latent productivity.
The difference is that the planner does not treat markup pricing as a technological constraint. Instead, the planner allocates labor, capital services,
materials, and output so as to maximize household welfare subject only to feasibility, the production technology, and the same allocated firms. Formally, conditional on the producing firm set $A_(j,t)$ in each market, the planner chooses sequences
#math.equation(block:true, numbering: none, ${C_t, Y_(j,t), y_(j i,t), l_(j i,t), k_(j i,t), m_(j i,t)}_(t>=0)$)
to maximize
#math.equation(block:true, numbering: none, $W = E_0 sum_(t=0)^infinity beta^t U(C_t,L_t),$)<eq:welfare_objective>
subject to the household aggregator @eq:agg_ces, the market bundles @eq:market_ces, the resource constraint @eq:resource, the production technology @eq:production, and the aggregate labor and capital constraints @eq:labor_market and @eq:capital_market. Let $lambda_t^L$ and $lambda_t^K$ denote the multipliers on the aggregate labor and capital constraints, and let
#math.equation(block:true, numbering: none, $psi_(j i,t) equiv (partial Q_t) / (partial y_(j i,t))$)<eq:planner_price>
denote the planner's marginal contribution of one additional unit of firm output to the gross composite bundle; since $C_t = Q_t - M_t - I_t$, this is also the marginal contribution to consumption holding materials and investment fixed. The planner's allocation conditions are then
#math.equation(block:true, $U_C(C_t,L_t) psi_(j i,t) (partial y_(j i,t))/(partial l_(j i,t)) = lambda_t^L,$)<eq:planner_labor_foc>
#math.equation(block:true, $U_C(C_t,L_t) psi_(j i,t) (partial y_(j i,t))/(partial k_(j i,t)) = lambda_t^K,$)<eq:planner_capital_foc>
#math.equation(block:true, $U_C(C_t,L_t) psi_(j i,t) (partial y_(j i,t))/(partial m_(j i,t)) = U_C(C_t,L_t).$)<eq:planner_materials_foc>
The planner equalizes the social marginal value of labor and capital across product lines, accounting for heterogeneity in scalability and productivity, and uses materials until their marginal product equals the value of the composite diverted from consumption. Because $psi_(j i,t)$ comes from the same demand system as the decentralized equilibrium, planner and market share a common inverse-demand structure and differ only in the markup wedge in pricing#footnote[@sec:planner_appendix gives $psi_(j i,t)$ explicitly and maps it through the nested CES structure.].

#v(0.3cm)

*Regime invariance and the welfare scale.* Every allocation considered below is solved under the same two normalizations, $P = 1$ and $L = 1$, and at the same
required return @eq:euler_R. Labor is therefore pinned across regimes while capital is free in both the market and the planner allocation: by
@eq:factor_aggregates, the aggregate capital stock is whatever firms demand at $R$. Lifetime welfare $W^x$ of an allocation $x$ is the household objective evaluated at that allocation, and the consumption-equivalent gain from moving from the market (ME) to the planner (PE) allocation is the scalar $lambda$ that, applied to market consumption in every period, raises market welfare to $W^(P E)$. The labor-disutility weight $chi$ is held at its baseline market value in every regime, and since $L = 1$ in every regime, the disutility term is common, such that with log utility over consumption the gain reduces exactly to
#math.equation(block:true, $1 + lambda = C^(P E) / C^(M E).$)<eq:lambda_is_C_ratio>
Because the planner keeps the producing firm sets $A_j$ of the market#footnote[Because the static cross section has exogenous firm counts, the planner does not re-select firms, so a full entry counterfactual is left outside the welfare exercise. The rationale behind this omission is the #cite(<edmond2023costly>, form:"prose") finding that the entry wedge is quantitatively small relative to the markup-level and misallocation channels.], $lambda$ measures the welfare cost of markup pricing at given producers, which I quantify for the calibrated economy in @sec:welfare and trace back there to the margins of the allocation that markups distort.
#v(0.5cm)
= Quantification

#v(0.5cm)
== Pooled-Market Steady State<sec:steady_state>
#v(0.2cm)

*Inner-market and Pooled aggregation block.* I now characterize the steady state as a nested system. Time subscripts are dropped, the aggregate price index is normalized to $P = 1$, and the distorted market steady state is written under the labor normalization $L = 1$. The resulting equilibrium is then used to pin down the labor-disutility weight $chi$, after which the market and planner steady states can be compared in common units. Conditional on sector $j$, the exogenous firm count $n_j$, and the realized firm draws $(alpha_(j i), nu_(j i))_(i in A_j)$, the steady-state market equilibrium solves jointly for firm quantities, prices, shares, markups, profits, and input demands through @eq:firm_demand, @eq:emx_markup, @eq:pricing, @eq:profits, and @eq:input_demand. The $alpha$ draw comes from $F^alpha$ and the productivity draw $nu$ follows @eq:nu_ladder and @eq:nu_pareto. Sector outcomes aggregate through the exposure-weighted CES bundle in @eq:agg_ces, with equal $omega_j$ across markets in the quantitative implementation. Consumption is $C = Q - M - delta_K K$, net of materials and replacement investment.


#v(0.3cm)

*Capital and factor-market block.* Aggregating the conditional input demands @eq:input_demand over the continuum of markets gives aggregate factor use as a function of aggregate variable cost $T C equiv integral_0^1 sum_(i in A_j) T C_(j i) d j$ alone. Only the fraction $phi.alt_v$ of variable cost is a primary-factor payment; the remaining $1-phi.alt_v$ buys materials of the
composite at $P = 1$:
#math.equation(block:true, $K = frac(a phi.alt_v "TC", R), quad L = frac((1-a) phi.alt_v "TC", w), quad M = (1-phi.alt_v) "TC", quad C = Q - M - delta_K K.$)<eq:factor_aggregates>
With $R$ fixed at @eq:euler_R, @eq:factor_aggregates makes the aggregate capital stock a residual: whatever quantity firms demand at the Euler rate is supplied. The anchored steady-state equilibrium is therefore a three-dimensional fixed point in $(w, X, underline(nu))$, where $X$ is composite expenditure per market and $underline(nu)$ is the productivity scale. The three residuals are the price normalization $P = 1$, the labor normalization $L = 1$, and the median active-firm output condition $tilde(y) = hat(y) equiv 1$. The rental rate is not among the unknowns.

#v(0.3cm)

*Normalization and steady-state welfare block.* At each candidate parameter vector, the distorted market steady state is anchored by the normalizations $P = 1$ and $L = 1$, together with the baseline anchoring condition that the median active-firm output equals the common anchor, $tilde(y) = hat(y) equiv 1$, imposed as one equilibrium condition of the baseline market solve#footnote[This condition pins the productivity scale $underline(nu)$ (the Pareto lower bound) as a by-product of its level normalization.]. Conditional on assigned parameters and on the anchored productivity distribution, the market solve pins down the wage-consumption ratio and therefore the labor-disutility weight $chi = w / C,$
using the market labor-leisure condition under $L = 1$ and $C = Q - M - delta_K K$. Once this baseline calibration step is complete, the planner and market steady states are re-solved in common units while holding the primitive objects fixed. The planner and market allocations therefore share the same $(beta, delta_K, a, phi.alt_v, hat(y), alpha_(j i), nu_(j i), N, xi, gamma, eta, macron(rho), H)$ and differ only through the presence or absence of the aggregate-markup and dispersion distortions. This is what makes the later welfare comparison interpretable as a comparison across allocations rather than across calibrations.

#v(0.3cm)

*Frozen-anchor protocol.* The anchor $hat(y)$ and all firm draws $(alpha_(j i), nu_(j i))$ are fixed at the baseline and held frozen across the planner solve and every intermediate allocation of @sec:welfare_decomp: technology is then an ordinary allocation-independent object, so the planner faces literally the same $T C_(j i,t)(y)$ as the market and re-anchoring under the planner is structurally impossible. The recalibrated economies of @sec:welfare_baseline keep the same definition of the anchor, the median active-firm output, but solve it on their own draws, such that its value changes with each economy, while the sorting arrangements of @sec:sorting keep the baseline value. This follows the logic of the normalized-CES literature #cite(<klump2000economic>), in which technology families are normalized to coincide at a common benchmark before varying curvature parameters. The fixed-$hat(y)$ construction applies the same identification principle to heterogeneous returns to scale: all technologies agree on marginal cost at the reference output, so differences in $alpha$ affect only the curvature of marginal cost away from that operating point. I go beyond #cite(<klump2000economic>, form: "prose") by arguing that the benchmark must also be economically sensible, and the median is preferable to the sales-weighted mean in skewed distributions. Active-firm output is heavily right-skewed, so the sales-weighted mean is pulled deep into the tail of that distribution, exceeding the output of roughly 99.6% of active firms in the baseline draw, such that anchoring there would make $alpha_(j i)$ act, for nearly the entire firm population, as a mechanical penalty for being smaller than a handful of superstar firms. The median sidesteps this by construction, since it splits the active-firm population evenly on either side of $hat(y)$ regardless of how fat the right tail is, so $alpha_(j i)$ is disciplined by curvature around a typical operating scale rather than by distance from the largest incumbents.

#v(0.5cm)
== Data and Measurement<sec:data>
#v(0.2cm)

*Sample, Sales, and Markups.* I use Spanish Orbis firm-years from 2015--2019, with monetary values expressed in 2019 euros. The sample excludes NAICS2 codes 52/53/62/81 following #cite(form: "prose", <hubmer2025scalable>), 92/99 for being out of scope, and 55 for consisting of holding companies and head offices. Manufacturing (31--33), retail (44--45), and transportation (48--49) are grouped into single cells, leaving 14 NAICS2-based cells. The sales and markup sample contains 2,000,342 firm-years from 568,920 firms.

To measure the upper tail of firm size, I average each firm's real sales over the window and divide by the mean of these firm means within its NAICS2 cell. I then pool the normalized sales and estimate the tail index $zeta$ as the negative slope of $ln(r - 1/2)$ on log normalized sales, using separately the largest 5% and 2% of firms ranked by sales. The resulting indices are $1.1079$ and $1.1451$, respectively.

Firm-level markups follow the production-function approach of #cite(<de2020rise>, form: "prose"). Each firm-year markup is the cell-year estimated variable-input elasticity times the firm's sales divided by its materials and staff costs. I weight these markups by materials and staff costs plus capital user cost within each year, then average the five annual values. The resulting aggregate markup target is $1.1815$.


#v(0.3cm)

*Scalability and Firm Size.* The distribution $F^alpha$ is not calibrated. It is built once in the empirical pipeline and the solver samples draws from its support. The firm-level object is the gross-output returns-to-scale estimate,
#math.equation(block:true,numbering: none, $alpha_(j i) = epsilon^m_(j i) + epsilon^l_(j i) + epsilon^k_(j i),$)
where $epsilon^m$, $epsilon^l$ and $epsilon^k$ are the GNR gross-output materials, labor and capital elasticities, estimated separately for each NAICS2 sector on the Orbis firm-year panel as in #cite(<hubmer2025scalable>, form: "prose")#footnote[The same estimates deliver the value-added weight $phi.alt_v = 1 - macron(epsilon)^m slash macron(alpha) = 0.517$ of @tab:param_ss.].

Firm-years are collapsed to one window mean per firm, so $F^alpha$ is a cross section over permanent scalability types, pooled across retained sectors rather than binned by NAICS2. The pooled distribution is winsorized at the 1st and 99th percentiles and discretized to 500 equal-mass nodes. The resulting model-facing support has mean $0.974$, standard deviation $0.094$, and 32.8% of its mass at $alpha >= 1$. In the model, scalability is bounded by the CES markup floor, $alpha_(j i) < gamma slash (gamma - 1)$, since cost minimization requires returns to scale not to exceed the markup #cite(<ruzic2023returns>), and above it the anchored equilibrium has no solution. Draws therefore come from $F^alpha$ truncated at $0.99 gamma slash (gamma - 1)$, that is, from the support nodes below this bound with their mass renormalized.#footnote[At the calibrated $gamma$ the bound is $1.134$, which excludes 5.2% of the support mass.]

#figure(
  block(width: 100%)[
    #image("figures/orbis/fig_rts_by_size_orbis.pdf", width: 90%)
    #align(left, text(size: 7.5pt)[_Note._ Mean returns to scale (left) and mean output elasticity and revenue share of intermediate inputs (right) by sales percentile for 1,391,061 Orbis firm-years of 2015--2019. The pooled curve ranks firm-years across industries within each year and demeans returns to scale by the full-sample mean. The within-industry curves rank and demean firm-years within NAICS2 industry-year cells. The plotted within-industry RTS gap is $0.071$. The $0.046$ calibration target instead ranks firms by mean sales within NAICS4 markets and uses mean RTS per firm. $F^alpha$ also uses firm-mean RTS.])
    #v(0.5em)
  ],
  caption: [Returns to scale and intermediate inputs by sales percentile.],
)<fig:alpha_by_size>

@fig:alpha_by_size plots mean scalability against firm-year sales percentiles, both pooled across industries within each year and within NAICS2 industry-year cells, in twenty equal-width bins for the 1,391,061 firm-years behind $F^alpha$. Returns to scale are U-shaped in size: firms in the bottom 5% of their industry's sales distribution have $alpha$ about $0.038$ above the industry mean, the profile falls to about $0.017$ below it between the 50th and 75th percentiles, and it then rises steeply to $0.074$ above the mean for the top 5%. Ranking firms in the pooled sample gives nearly the same curve, so the profile is not a composition effect across industries. As can be seen in the right panel, the materials output elasticity rises from $0.170$ below the industry mean for the smallest firms to $0.162$ above it for the largest, while the materials revenue share rises from $0.094$ below to $0.096$ above. The labor elasticity falls with size and lifts the left arm of the U. This is the shape #cite(<hubmer2025scalable>, form: "prose") find for U.S. and Canadian manufacturing, where returns to scale are also U-shaped in revenue and the steep increase at the top is also driven by the intermediate input elasticity, albeit with a less pronounced left arm than in the Spanish data.


In the model, $alpha$ alone does not deliver a size ordering: at the anchor $hat(y)$ marginal cost is independent of $alpha$, and a higher $alpha$ lowers marginal cost above the anchor but raises it below, so scalability by itself spreads firms around $hat(y)$ rather than ranking them by size. Their position in the size distribution therefore depends on the alignment between scalability and productivity that $macron(rho)$ governs in @eq:nu_ladder, which I calibrate in @sec:calibration to the gap in mean $alpha$ between the top 5% and the bottom 50% of firms by sales measured on firm means within NAICS4 markets following #cite(<hubmer2025scalable>, form: "prose"), which for the Spanish Orbis data is around $0.046$.



#v(0.5cm)
== Calibration<sec:calibration>
#v(0.2cm)

*Parameters and targets.* In the $(alpha,nu)$-economy baseline the calibrated vector is $(xi, N, gamma, macron(rho))$, which targets four pooled scalars, namely the aggregate cost-weighted markup $mu_("cw")^("data") approx 1.18$, the firm-sales tail indices $zeta$ at the top 5% and top 2%, and the returns-to-scale gap between the top 5% and the bottom 50% of firms by sales. The within-market elasticity $gamma$ controls the common substitution environment and helps pin the markup level through the markup target, whose model analogue is #math.equation(numbering:none,block:true,$mu_("cw")^("model") = integral_0^1 sum_(i in A_j) lambda^c_(j i) mu_(j i) d j, quad lambda^c_(j i) equiv (T C_(j i))/(integral_0^1 sum_(h in A_j) T C_(j h) d j)$)
, where $lambda^c_(j i)$ is firm $i$'s share of aggregate variable cost. The across-market elasticity $eta$ governs cross-sector substitution and is fixed at $1.2$, a value within the range of across-market elasticities in the literature #cite(<atkeson2008pricing>) #cite(<deloecker2021quantifying>) #cite(<edmond2015competition>). The Pareto tail index $xi$ of the fundamental productivity $nu$ and the Poisson mean $N$, which sets the mean number of firms in a sector, jointly discipline the upper tail of firm sales through the two tail indices, a choice motivated by the firm-size evidence of #cite(<axtell2001zipf>, form: "prose") and #cite(<gabaix2009power>, form: "prose"), and by the Zipf calibration in #cite(<di2012country>, form: "prose"). On the data side, these indices are OLS estimates on firm-mean sales over 2015--2019, normalized by the NAICS2 mean of firm means, and the model applies the same estimator to pooled active-firm sales. Orbis concentration measures are avoided as calibration targets, since #cite(<kalemli2024construct>, form: "prose") show that measured industry concentration depends on the representativeness of the firm sample, and concentration measures in general tend to be highly dependent on market definition, which is by itself subjective and can lead to severely different results#footnote[See #cite(<edmond2015competition>, form: "prose"), #cite(<deloecker2021quantifying>, form: "prose") among others.].

The rank-copula parameter $macron(rho)$ sets the strength of the $alpha$--$nu$ dependence and is calibrated to the within-market returns-to-scale gap of $0.046$, the mean demeaned $alpha$ of the top 5% minus the bottom 50% of firms by sales within each NAICS4 market. This follows #cite(<hubmer2025scalable>, form: "prose"), who use the same top-5%-versus-bottom-50% RTS-gap statistic to help discipline the relationship between TFP and returns to scale in their calibration. They also meet the problem that the joint distribution of returns to scale and TFP cannot be read off the production-function estimates#footnote[Rescaling output and inputs by a factor $s$ shifts estimated log TFP $ln hat(z)_(j i)$ by the firm-specific amount $(1 - alpha_(j i)) ln s$, such that relative TFP across firms has no cardinal interpretation.], and choose to calibrate the correlation between the two internally rather than imposing the estimated one. The productivity scale $underline(nu)$ is not part of the calibrated vector, being solved by the baseline anchoring condition $tilde(y) = hat(y) equiv 1$, which centers the $alpha$-gradient at the operating scale rather than fighting the returns-to-scale gap moment. Objects treated as externally disciplined are the pooled scalability distribution $F^alpha$, the firm-level gross-output returns to scale $alpha_(j i)$, the value-added weight $phi.alt_v$, and the common capital share $a = 1 slash 3$ taken from #cite(<edmond2023costly>, form:"prose"). See @tab:param_ss for an overview of all parameters and their final values.

#v(0.5cm)

*Calibration Results.* As shown in @tab:calib_fit, all four moment targets are hit to within $0.015%$, with the largest deviation being for the top-2% firm-sales tail index, around $0.014%$ off. As for the resulting parameter values, the within-market elasticity $gamma = 7.88$ implies a common CES markup floor of $1.145$. Thus, $gamma$ supplies most of the average markup wedge through the common CES floor ($80.1%$), with the remaining $19.9%$ coming from endogenous share-driven market power.


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

      [$gamma$ (within-market elast.)], [7.88], [$mu_("cw")$], [1.1815], [1.1816], [0.001],
      [$xi$ (Pareto tail of $nu$)], [9.50], [Sales tail index $zeta$, top 5%], [1.1079], [1.1078], [0.008],
      [$N$ (Poisson mean firms)], [604.0], [Sales tail index $zeta$, top 2%], [1.1451], [1.1452], [0.014],
      [$macron(rho)$ (rank copula $alpha$--$nu$)], [$-0.166$], [RTS gap, top 5% $-$ bottom 50%], [0.0460], [0.0460], [0.007],
      [$eta$ (across-market elast.)], [1.2], [fixed, not calibrated], [], [], [],

      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ The four calibrated parameters are solved jointly, while $eta$ is fixed at $1.2$. The row pairing is for presentation: both sales-tail indices jointly discipline $xi$ and $N$. Dev. is the absolute deviation of the model moment from its target, in percent of the target. Full parameterization in Appendix @tab:param_ss.])
    #v(0.5em)
  ],
  caption: [Calibrated parameters and model fit.],
)<tab:calib_fit>

Moving on to the Pareto tail parameter $xi = 9.50$, note that it is a tail on productivity $nu prop z^(1/alpha)$ and not on the TFP coefficient $z$, whose implied distribution is a mixture of Paretos with indices $xi "/" alpha_(j i)$, such that $xi$ is not directly comparable with tail parameters on TFP under constant returns to scale.

Matching the two firm-sales tail indices in the Cournot economy yields a Poisson mean of $N=604$. This remains the mean number of firms competing within a market, but the calibration identifies it from the pooled sales tails rather than an observed firm count per market. Changing $N$ changes how productivity draws translate into market shares and the shape of the upper sales tail, so the top 5% and top 2% indices jointly distinguish its effect from that of $xi$. In the calibrated economy, many firms hold small shares and charge markups close to the common CES floor, while a few reach the upper tail. Since all allocated firms produce, $N$ includes the small-firm fringe.#footnote[As a descriptive model diagnostic, the sales-weighted mean of $1 "/" "HHI"$ across simulated markets is 51.1, compared with 603.7 active firms per market on average. This statistic is computed from model sales shares and is not a calibration target.]

Finally, the rank-copula parameter $macron(rho) = -0.166$ corresponds to corr$(nu, alpha) approx -0.14$ in the calibrated economy. Negative sorting places some high-$alpha$ firms with low-$nu$ firms at the small end of the sales distribution. Among firms that reach output above the anchor, higher $alpha$ lowers marginal cost as they expand, allowing high-$alpha$ firms to appear again at the top. This mechanism is consistent with the U-shaped RTS profile in @fig:alpha_by_size. The calibration directly matches the separately constructed top-versus-bottom RTS gap rather than the full profile. #cite(<hubmer2025scalable>, form: "prose") also obtain a negative calibrated correlation between TFP and returns to scale.


#v(0.5cm)
= Welfare Analysis<sec:welfare>

#v(0.5cm)
== Welfare Cost of Markups<sec:welfare_baseline>
#v(0.2cm)

*Markup Distribution and Welfare Cost.* The calibrated $gamma = 7.88$ implies a common CES floor of $mu_min = gamma / (gamma - 1) approx 1.145$. Against the fitted aggregate markup of $1.182$, this floor contributes $80.1%$ of the average wedge above one and the share-driven component of @eq:emx_markup contributes $19.9%$. Weighting firms by cost, the 25th, 50th and 75th percentiles of firm markups, $1.146$, $1.148$ and $1.175$, lie close to the floor. The calibration matches sales tails from a broad Spanish Orbis sample. As mentioned above, its $N approx 604$ includes a large fringe of small active firms without a selection margin to drop them, leaving many Cournot shares small and markups near the floor. Firms with larger shares still form a thin upper tail. At the 90th and 99th percentiles, firm markups are $1.266$ and $1.529$, against $1.221$ and $1.331$ for sector markups, defined as the cost-weighted mean markup within each sector and ranked by sector cost. Therefore, an aggregate markup of $1.18$ can lower welfare through two rather different objects, namely a common wedge that almost every firm charges and a thin tail of large firms that charge far more, and which of the two dominates the welfare impact depends on how the planner would move production across these firms. Removing markups altogether raises steady-state consumption by $lambda = 32.3%$ in the calibrated economy, with the planner producing a gross composite $Q$ that is $72.8%$ larger than in the market, with $114.1%$ more capital (bottom row of @tab:welfare_hetero, Panel A). Since the distribution of markups comes from the heterogeneity of firms, the first question is how much of this cost is due to firms differing in productivity, in scalability or in both.

#v(0.3cm)
*Counterfactual Economies.* To isolate the role of each dimension of heterogeneity, I compare the calibrated economy with three counterfactual economies, each recalibrated on its own free parameters to the targets it can reach, such that the comparison holds observables fixed rather than parameters. The $nu$-economy holds RTS common at the harmonic mean of $F^alpha$ and recalibrates $(xi, N, gamma)$, while the $alpha$-economy holds productivity common and the homogeneous economy holds both common, which leaves only $gamma$ to recalibrate. The calibration results and moment fit are displayed in @tab:recal_calib in the Appendix. All four economies match $mu_("cw")$, whereas only the baseline also matches the two Zipf indices and the $nu$-economy nearly matches them, since a bounded RTS distribution generates no Pareto tail in sales.#footnote[With common $nu$, size differences come only from $alpha$, and the capped, data-disciplined $F^alpha$ keeps every firm away from the singularity at $gamma slash (gamma - 1)$ where size diverges, so sales have no Pareto tail. #cite(<hubmer2025scalable>, form: "prose") obtain one only by letting returns to scale approach that bound.] Without a fat tail no firm reaches a sizeable share, such that markups collapse onto the CES floor and $gamma$ is pinned by the near-atomistic limit $gamma slash (gamma - 1) approx mu_("cw")$, which is why the homogeneous and $alpha$-economies share $gamma = 6.556$. As for the $nu$-economy fit, without heterogeneity in $alpha$, the Zipf indices must come from $nu$ alone, which requires a fatter productivity tail ($xi = 5.05$ against $9.50$). Within-market shares, and so $mu_("cw")$, then rise, and the fit offsets this by spreading sales over more firms per market ($N = 1998$).

#v(0.3cm)
#figure(
  block(width: 100%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (3.8fr, 1.3fr, 1.35fr, 1.4fr),
      column-gutter: 0.22cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right),
      inset: (x: 0pt, y: 3pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.header(
        table.cell(rowspan: 2, align: left + bottom)[*Panel A. Economy*],
        table.cell(colspan: 3, align: center)[#text(size: 8.5pt)[*Planner vs. market (% change)*]],
        table.hline(y: 1, start: 1, end: 4, stroke: 0.4pt),
        table.cell(align: right)[$Q$],
        table.cell(align: right)[$K$],
        table.cell(align: right)[*$C$ (welfare $lambda$)*],
      ),
      table.hline(stroke: 0.4pt),
      [Homogeneous Economy], [32.39], [56.42], [10.77],
      [$nu$-Economy], [36.46], [61.28], [14.25],
      [$alpha$-Economy], [39.13], [67.76], [12.96],
      [*$(alpha,nu)$-Economy (Calibrated Baseline)*], [*72.78*], [*114.12*], [*32.04*],
      table.hline(stroke: 0.7pt),
    )
    #v(0.15cm)
    #table(
      columns: (3.8fr, 1.3fr, 1.35fr, 1.4fr),
      column-gutter: 0.22cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right),
      inset: (x: 0pt, y: 3pt),
      stroke: none,

      table.header(
        table.cell(rowspan: 2, align: left + bottom)[*Panel B. Component*],
        table.cell(colspan: 3, align: center)[#text(size: 8.5pt)[*Contributions (pp; shares in parentheses)*]],
        table.hline(y: 1, start: 1, end: 4, stroke: 0.4pt),
        table.cell(align: right)[$Q$],
        table.cell(align: right)[$K$],
        table.cell(align: right)[*$C$ (welfare $lambda$)*],
      ),
      table.hline(stroke: 0.4pt),
      [Common level (both homogeneous)], [32.39 (44.5%)], [56.42 (49.4%)], [10.77 (33.6%)],
      [Productivity dispersion, $alpha$ held common ], [4.07 (5.6%)], [4.86 (4.3%)], [3.48 (10.9%)],
      [RTS dispersion, $nu$ held common ], [6.74 (9.3%)], [11.34 (9.9%)], [2.19 (6.8%)],
      [*Joint heterogeneity interaction*], [*29.58 (40.6%)*], [*41.50 (36.4%)*], [*15.60 (48.7%)*],
      table.hline(stroke: 0.4pt),
      [*Total (Calibrated Baseline)*], [*72.78 (100%)*], [*114.12 (100%)*], [*32.04 (100%)*],
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ Panel A compares each economy's planner with its own market, where $Q$ is the gross composite @eq:agg_ces, $K$ is capital and the change in consumption $C$ equals the consumption-equivalent welfare gain $lambda = C^(P E) / C^(M E) - 1$ by @eq:lambda_is_C_ratio, all in percent. Panel B splits the calibrated gains into percentage-point contributions (shares in parentheses). Each counterfactual economy is recalibrated to match $mu_("cw")$, with the $nu$-economy (common scalability at the harmonic mean of $F^alpha$) refitting $(xi, N, gamma)$ and also nearly matching the Zipf indices of sales, while the $alpha$- and homogeneous economies (common productivity) refit only $gamma$, since a bounded scalability distribution generates no Pareto tail. All economies use $M = 5000$ markets and their own $chi$. The joint interaction is distinct from the sorting effect in @tab:sorting.])
    #v(0.5em)
  ],
  caption: [Productivity, scalability, and the welfare cost of markups.],
)<tab:welfare_hetero>
#v(0.3cm)

*Which Heterogeneity Matters.* The most informative comparison is between the $nu$-economy and the calibrated baseline, which match the same markup level and nearly the same sales tails with almost the same $gamma$ ($7.78$ against $7.88$). Adding the observed heterogeneity in scalability on top of the $nu$-economy more than doubles the welfare cost of markups, from $14.3%$ to $32.3%$, while the homogeneous economy, at $10.8%$, gives the cost of a common markup of $1.18$ alone (@tab:welfare_hetero, Panel A). The output and capital responses of the planner follow a different ranking, as the $alpha$-economy expands the gross output by $39.1%$ and capital by $67.8%$, slightly more than the $nu$-economy ($36.5%$ and $61.3%$) despite its smaller welfare gap, while the baseline expands both by about twice as much as the homogeneous economy ($72.8%$ and $114.1%$, against $32.4%$ and $56.4%$). In every economy capital grows faster than output and output faster than consumption (equivalent to $lambda$ as per @eq:lambda_is_C_ratio), since the larger capital stock and materials use of the planner are paid for out of $Q$, with $C = Q - M - delta_K K$ in steady state, which suggests that heterogeneity in RTS raises the welfare cost mostly through how much more the planner would produce, a point to which I return in @sec:welfare_decomp. 

Moving on to Panel B, the calibrated gain is split into the gain of the homogeneous economy, the gains of the $nu$- and $alpha$-economies over it and a residual joint interaction, such that the four contributions sum to $lambda$ by construction#footnote[The contributions are $lambda^"hom"$, $lambda^nu - lambda^"hom"$, $lambda^alpha - lambda^"hom"$ and $lambda^(alpha nu) - lambda^nu - lambda^alpha + lambda^"hom"$. Since every economy is recalibrated, each contribution compares economies that match the same targets, rather than switching one dimension off at given parameters.]. Each individual heterogeneity margin contributes little on its own, with $3.48$ percentage points from productivity dispersion and $2.19$ from scalability dispersion. As in Panel A, RTS heterogeneity has the biggest contribution to the capital and therefore the $Q$ gap. Likewise, productivity dispersion ends up having the biggest welfare role out of the two again, potentially due to the lack of an expressive Pareto tail under RTS dispersion only. The joint interaction of the two accounts for $15.60$ percentage points, or $48.7%$ of the calibrated gain, such that almost half of the welfare cost of markups under the calibrated baseline arises only when firms differ along both dimensions at once. Understanding what is behind this interaction effect requires splitting a welfare gain into the channels through which wedges lower consumption, which I turn to next.

#v(0.5cm)
== Decomposing the Welfare Cost<sec:welfare_decomp>
#v(0.2cm)

*Welfare Setup.* To investigate the components behind the interaction effect, I define and work with the log of the gain $lambda$ of @eq:lambda_is_C_ratio,
#math.equation(block:true, $Delta equiv ln(1 + lambda) = ln C^(P E) - ln C^(M E),$)<eq:delta_def>
which is additive across any chain of intermediate allocations between the market and the planner, such that each decomposition below is an exact identity rather than a local approximation. Every intermediate allocation is a steady state solved on the same draws $(alpha_(j i), nu_(j i))$, the same active firm sets and the same frozen anchor $hat(y)$ as the market and the planner. Additionally, it shares their normalizations $P = 1$ and $L = 1$ and the market value of $chi$, such that allocations differ only in pricing and in the constraints they place on aggregate inputs.

The first decomposition asks whether markups lower welfare through their common level or through their dispersion across firms, as is common in the literature #cite(<baqaee2020productivity>) #cite(<edmond2023costly>) #cite(<de2020rise>). To decompose the total effect into these two components, I introduce the uniform-markup allocation ($U$), where every firm charges the common cost-weighted average markup $macron(mu) = mu_("cw")^("model")$ of the market @eq:mu_cw_identity, such that the aggregate wedge is preserved by construction and only its dispersion across firms is removed. The decomposition then splits the gain into a _dispersion_ component, from the market to $U$, and a _level_ component, from $U$ to the planner, such that
#math.equation(block:true, $Delta = Delta_"disp" + Delta_"level", quad Delta_"disp" = ln(C^U / C^(M E)), quad Delta_"level" = ln(C^(P E) / C^U).$)<eq:markup_decomp>

The second decomposition asks instead whether the planner gains by reallocating the inputs (i.e. fixing misallocation) that the market already uses or by using more of them, in the spirit of #cite(<hsieh2009misallocation>, form: "prose"). The dispersion of revenue productivity, which #cite(<hsieh2009misallocation>, form: "prose") use to measure distortions, no longer maps into misallocation once returns to scale differ, since undistorted firms then differ in revenue productivity as well #cite(<ruzic2023returns>). I therefore compute the misallocation component from the counterfactual allocation itself. To explore these different components, I introduce two constrained planners, both pricing at marginal cost. The fixed-input planner ($P E_I$) holds aggregate capital and materials at their market levels, $K = K^(M E)$ and $M = M^(M E)$. Therefore, the rental rate $R$ rises above the Euler rate @eq:euler_R and a shadow materials price $P_"mat"$ takes the place of $P = 1$ inside the unit cost @eq:omega_gross, each adjusting until aggregate demand for its input returns to the market level. Since both prices enter the unit cost of every firm, this re-prices the whole cross section rather than relabeling the planner allocation. Finally, the fixed-capital planner ($P E_K$) frees materials at $P = 1$ while still holding capital at $K^(M E)$, and the input decomposition splits the same gain into _misallocation_, _materials_ and _capital_ components, such that
#math.equation(block:true, $Delta = Delta_"misalloc" + Delta_"mat" + Delta_"cap",$)<eq:input_decomp>
#math.equation(block:true, numbering: none, $Delta_"misalloc" = ln(C^(P E_I) / C^(M E)), quad Delta_"mat" = ln(C^(P E_K) / C^(P E_I)), quad Delta_"cap" = ln(C^(P E) / C^(P E_K)).$)

Note that the two decompositions are not nested, as the first cuts the gain by which wedges are removed and the second by which inputs are allowed to move. Both, however, sum exactly to the same $Delta$, such that the share of each component in $Delta$ is well defined and the shares add up to one within each decomposition.

#v(0.3cm)
*From the Homogeneous Economy to the Baseline.* In @fig:recal_channels, the welfare gain of each economy is stacked by both decompositions, moving from the homogeneous economy to the calibrated baseline in the steps of Panel B of @tab:welfare_hetero. The homogeneous economy has $Delta = 0.102$, all of it level, since its markups are uniform, and mostly capital ($0.080$), with the remaining $0.022$ coming from materials. Productivity dispersion then adds $0.031$, essentially all of it through dispersion ($0.032$) in Panel A and mostly through misallocation ($0.022$) in Panel B, whereas RTS dispersion adds $0.020$, all of it through the level and mostly through capital ($0.018$), with no dispersion component at all, since the markups of the $alpha$-economy are near uniform as well, as discussed above. Finally, the interaction contributes nearly half ($0.125$) of the final calibrated $Delta = 0.278$, and $70%$ of it comes through dispersion ($0.087$) in Panel A, while capital takes $58%$ of it ($0.073$) under the input decomposition in Panel B.

The calibrated bar then shows the composition of the full gain. The dispersion component accounts for $0.119$ of $Delta$, or $43%$, with the remaining $0.159$ coming from the common level, while the input decomposition attributes $0.054$ to misallocation, $0.048$ to materials and $0.176$, or $63%$, to capital. That is, reallocating the inputs that the market already uses would raise consumption by $5.5%$. This is the counterpart of the fixed-input gains of #cite(<hsieh2009misallocation>, form: "prose"), who find manufacturing TFP gains of $30$–$43%$ from equalizing revenue productivity in the United States, but it is much smaller, since only the dispersion created by markups is removed here, while their measure captures every source of dispersion in revenue productivity. Most of the gain, however, comes from the planner accumulating capital at the Euler rate, which amplifies fixed-input gains as in their exercise, and does so far more here, since the planner also removes the common markup level that equalizing revenue productivity leaves in place. #cite(<restuccia2008policy>, form: "prose") draw a similar distinction, holding aggregate capital fixed to isolate the reallocation effect of distortions, and find that letting capital adjust adds a further output loss through lower accumulation. Since all comparisons are across steady states, the capital component ignores the consumption forgone along the transition to a capital stock more than twice as large, and $lambda$ should be read as an upper bound on the gain net of accumulation.

#v(0.3cm)
#figure(
  block(width: 100%)[
    #align(center, image("figures/orbis/fig_recalibrated_channels_orbis_zipf.pdf", width: 100%))
    #align(left, text(size: 7.5pt)[_Note._ Bars walk along the welfare gain $Delta = ln(1 + lambda)$ from the homogeneous economy to the calibrated baseline in the steps of Panel B of @tab:welfare_hetero, namely common level ($Delta^"hom"$), productivity dispersion ($Delta^nu - Delta^"hom"$), RTS dispersion ($Delta^alpha - Delta^"hom"$) and the joint interaction ($Delta^(alpha nu) - Delta^nu - Delta^alpha + Delta^"hom"$). Each step is stacked by the components of one decomposition, which sum to $Delta$ exactly in every economy, namely the markup decomposition @eq:markup_decomp into dispersion ($M E -> U$) and level ($U -> P E$) in Panel A, and the input decomposition @eq:input_decomp into misallocation ($M E -> P E_I$), materials ($P E_I -> P E_K$) and capital ($P E_K -> P E$) in Panel B.])
    #v(0.5em)
  ],
  caption: [From the homogeneous economy to the calibrated baseline, by channel of the welfare gain.],
)<fig:recal_channels>

*Complementarity.* The two dimensions therefore act as complements. Productivity generates the fat tail of sales, and with it the large shares and high markups, while scalability shapes both how far a firm grows and how much output a given markup withholds from it, such that the cost of markups is largest where high productivity and high RTS meet. The markups of the $nu$-economy and the baseline make this point sharper. The cost-weighted variance of log markups is higher in the $nu$-economy than in the calibrated baseline, namely $0.0042$ against $0.0037$, and yet its dispersion component is about a quarter of the baseline ($0.032$ against $0.119$), such that the cost of dispersion depends on how scalable the firms carrying them are, and not only on how dispersed firm markups are.

A potential explanation comes from the marginal cost @eq:marginal_cost. Holding the sector aggregates fixed and abstracting from the feedback of shares on markups, a firm facing the within-market elasticity $gamma$ responds to a change in its markup with
#math.equation(block:true, $ epsilon_mu equiv frac(d ln y_(j i), d ln mu_(j i)) = - frac(gamma alpha_(j i), alpha_(j i) + gamma (1 - alpha_(j i))),$)<eq:markup_response>
which rises with $alpha_(j i)$, as the price elasticity of supply increases with returns to scale #cite(<ruzic2023returns>), and diverges as $alpha_(j i)$ approaches $gamma slash (gamma - 1)$, the bound below which the scalability draws are capped. Under constant returns, the elasticity $epsilon_mu$ defined in @eq:markup_response equals the calibrated $gamma = 7.88$ but rises to about $ epsilon_mu = 28$ at $alpha_(j i) = 1.1$, a value well inside the support of $F^alpha$. A given wedge thus withholds far more output when it sits on a highly scalable firm, and the RTS gap of $0.046$ that the calibration matches places such firms at the top of the sales distribution, where Cournot markups are highest. The convexity of @eq:markup_response in $alpha_(j i)$ could also explain why scalability dispersion alone raises the level component, since the average response to removing a common wedge grows with the spread of $alpha$. If the cost of markups is largest where high productivity meets high RTS, it should also depend on how the two are paired across firms, which the calibration disciplines through the RTS gap. The next natural question is then to ask how impactful the alignment/misalignment between productivity and RTS is for the welfare cost of wedges.

#v(0.5cm)
== Sorting and the Cost of Markups<sec:sorting>
#v(0.2cm)

*Sweep.* The interaction in @tab:welfare_hetero shows that the welfare cost of markups depends on productivity and scalability jointly, but it does not show how much it depends on the way the two are paired across firms. To answer this, I hold the calibrated parameters and both marginal distributions fixed and vary only the strength $macron(rho)$ of the rank copula @eq:nu_ladder. The experiment re-solves the economy for values of $macron(rho)$ between $-0.9$ and $+0.9$, keeping $(xi, N, gamma, eta)$, the scalability draws, the anchor $hat(y)$ and the baseline $chi$ at their calibrated values, such that the productivity marginal is preserved in distribution. Additionally, three arrangements permute the calibrated draws within each sector, namely perfect negative and positive rank pairings, labelled $macron(rho) = minus.plus 1$ in @tab:sorting, and a random reassignment that removes any dependence, all of which preserve both marginals exactly. Two features set this experiment apart from @tab:welfare_hetero. First, the arrangements are not recalibrated, such that they no longer match the RTS gap, the Zipf indices or $mu_("cw")$. The sweep therefore measures how the welfare cost responds to the pairing at given primitives, while @tab:welfare_hetero compares economies that agree on the observed moments. Second, the markup decomposition re-solves $U$ at each arrangement's own $mu_("cw")$, such that the level component absorbs any change in aggregate market power and the dispersion component captures the cost of markups differing across firms at a given level.

#v(0.3cm)
#figure(
  block(width: 100%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (2.3fr, 1fr, 0.8fr, 1fr, 1.1fr, 0.9fr, 0.9fr, 1fr),
      column-gutter: 0.2cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right, right, right, right),
      inset: (x: 0pt, y: 3pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      table.header(
        table.cell(rowspan: 2, align: left + bottom)[*Arrangement $macron(rho)$*],
        table.cell(rowspan: 2, align: right + bottom)[$"corr"(alpha, nu)$],
        table.cell(colspan: 3, align: center)[#text(size: 8.5pt)[*Market markups*]],
        table.cell(colspan: 3, align: center)[#text(size: 8.5pt)[*Planner vs. market (% change)*]],
        table.hline(y: 1, start: 2, end: 5, stroke: 0.4pt),
        table.hline(y: 1, start: 5, end: 8, stroke: 0.4pt),
        table.cell(align: right)[$mu_("cw")$],
        table.cell(align: right)[$"Var"(ln mu)$],
        table.cell(align: right)[$"corr"(ln mu, alpha)$],
        table.cell(align: right)[$Q$],
        table.cell(align: right)[$K$],
        table.cell(align: right)[*Welfare*],
      ),
      table.hline(stroke: 0.4pt),
      [Lowest: $macron(rho) = -1$], [$-0.92$], [1.147], [0.00], [--], [28.7], [49.9], [9.0],
      [Sweep: $macron(rho) = -0.60$], [$-0.54$], [1.149], [0.01], [$0.20$], [31.6], [54.0], [10.3],
      [Sweep: $macron(rho) = -0.40$], [$-0.35$], [1.157], [0.08], [0.37], [43.2], [71.3], [16.0],
      [*Calibrated* *$macron(rho) = -0.166$*], [*$-0.14$*], [*1.182*], [*0.37*], [*$0.48$*], [*73.3*], [*114.9*], [*32.3*],
      [*Random reassignment*], [*$approx 0.00$*], [*1.210*], [*0.72*], [*$0.51$*], [*104.7*], [*160.1*], [*50.3*],
      [Sweep: $macron(rho) = +0.30$], [$+0.25$], [1.271], [1.41], [$0.50$], [175.8], [264.2], [91.3],
      [Sweep: $macron(rho) = +0.60$], [$+0.48$], [1.324], [1.76], [$0.46$], [240.3], [361.2], [127.6],
      [Sweep: $macron(rho) = +0.75$], [$+0.60$], [1.339], [1.69], [$0.43$], [258.6], [388.8], [137.6],
      [Highest: $macron(rho) = +1$], [$+0.77$], [1.356], [1.54], [$0.34$], [284.9], [427.4], [152.3],
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ $Q$ is the gross composite @eq:agg_ces and $K$ capital, with materials expanding by the same percentage as capital. Changes compare planner and market steady states within each row at fixed labor and firm counts, such that the welfare column equals the consumption increase. Rows are indexed by the copula parameter $macron(rho)$, next to the realized corr$(alpha, nu)$, with $macron(rho) = plus.minus 1$ the perfect rank pairings obtained by within-sector permutation and random reassignment the average of two permutations. All rows keep the baseline anchor and $chi$ and use $M = 10000$ markets, against $M = 5000$ in @tab:welfare_hetero, which is why the calibrated row differs slightly from that table. Markups are over active firms and cost-weighted, with $"Var"(ln mu)$ multiplied by 100, and their correlation with $alpha$ is undefined at $macron(rho) = -1$, where all markups sit at the CES floor.])
    #v(0.5em)
  ],
  caption: [Sorting and the welfare cost of markups.],
)<tab:sorting>

*How Much the Pairing Matters.* The pairing of scalability and productivity moves the welfare cost of markups by more than any dimension of heterogeneity in @tab:welfare_hetero. As shown in @tab:sorting, $lambda$ rises from $9.0%$ under perfect negative pairing to $152.3%$ under perfect positive pairing#footnote[The perfect pairings, labelled $macron(rho) = minus.plus 1$, are boundary cases rather than plausible economies, and they serve to bound the welfare cost of markups at the calibrated shared marginals.], with random reassignment at $50.3%$. This echoes #cite(<restuccia2008policy>, form: "prose"), who show that idiosyncratic distortions are far more costly when they are correlated with establishment productivity than when they are not, although here the correlation between markups and scalability arises in equilibrium rather than being imposed. The mildly negative sorting of the calibrated economy ($-0.14$) keeps the cost well below that of independent assignment or even positive pairing. The response is clearly S-shaped, as can be visually confirmed in Panel A of @fig:sorting_curve in the Appendix, being flat under strongly negative sorting, steepest around the calibrated point and flattening again at the top, where the last copula point (corr$(alpha, nu) = 0.71$) lies marginally below the previous one and only the exact permutation reaches $152.3%$. Around the calibrated economy, $lambda$ doubles between a correlation of $-0.35$ and $-0.14$ (from $16.0%$ to $32.3%$) and rises by a further $18$ percentage points to random assignment, which makes $macron(rho)$, and the RTS gap that identifies it, a first-order determinant of the welfare cost under RTS and productivity heterogeneity. Note that concluding that sorting decreases welfare would be incorrect. In fact, Panel B of @fig:sorting_curve shows that sorting raises welfare in both allocations, as lifetime welfare of the market and of the planner both rise with corr$(alpha, nu)$. However, the planner rises faster, which widens the gap, and thus the welfare gain from going from market to planner. Pairing scalable with productive firms thus raises what the economy can produce, while markups withhold a growing share of it.

Sorting also creates market power. Since aligning scalability with productivity lets the most productive firms expand further, their shares grow and $mu_("cw")$ rises from $1.147$ to $1.356$ along the sweep, with the CES floor of $1.145$ unchanged. Markups also differ more across firms, as the variance of log markups in @tab:sorting rises almost fivefold from the calibrated point to $macron(rho) = +0.6$. Their correlation with scalability, by contrast, rises only up to random assignment, from $0.20$ at $macron(rho) = -0.6$ to about $0.5$, and falls to $0.34$ under perfect positive pairing. Part of the rise in $lambda$ must therefore come from a higher markup level, and part from markups differing more across firms. At the same time, the capital that the planner would add grows from $49.9%$ to $427.4%$ of the market stock along the sweep, with materials likewise expanding (@tab:sorting). This indicates that the cost may lie less in how the market allocates the inputs it uses than in the inputs it withholds altogether. The two decomposition strategies presented in @sec:welfare_decomp are applied here to quantify these different channels.


#figure(
  block(width: 100%)[
    #align(center, image("figures/orbis/fig_sorting_lenses_orbis_zipf.pdf", width: 100%))
    #align(left, text(size: 7.5pt)[_Note._ Each point is a solved economy plotted at its realized corr$(alpha, nu)$, as in @fig:sorting_curve, and the dashed lines mark the calibrated economy. The black line is the total steady-state gain $Delta = ln(1 + lambda)$, and the shaded components sum to it exactly. Panel A splits it by @eq:markup_decomp into dispersion ($M E -> U$) and level ($U -> P E$), with $U$ re-solved at each economy's own $mu_("cw")$. Panel B splits it by @eq:input_decomp into misallocation ($M E -> P E_I$), materials ($P E_I -> P E_K$) and capital ($P E_K -> P E$). Labor, firm sets, the anchor and $chi$ are fixed throughout, and areas between points are interpolated.])
    #v(0.5em)
  ],
  caption: [Decomposition of the welfare gain from the planner along the copula sweep.],
)<fig:sorting_decomp>
#v(0.3cm)

*Decomposing Sorting.* I reuse the welfare decomposition machinery from @sec:welfare_decomp, solving the intermediate planner allocations ($U,P E_I, P E_K$) at each arrangement. Since $U$ is re-solved at each arrangement's own $mu_("cw")$, the level component of the markup decomposition in Panel A of @fig:sorting_decomp absorbs the higher markup level, while the dispersion component captures their spread across firms, and the latter rises from zero under strongly negative sorting to $0.12$ at the calibrated point and to about $0.45$ at the top, around half of the total $Delta$. Moving on to Panel B, the misallocation component rises from zero to about $0.2$, or roughly a quarter of $Delta$, since a common markup leaves nothing to reallocate at fixed inputs, while capital remains the largest component throughout, such that sorting raises the cost of markups both by misplacing the inputs already in use and, above all, by holding back the expansion of scalable firms. Up to random assignment, both rises go along with a wider spread of markups and a growing correlation between markups and scalability in @tab:sorting. Beyond it, the correlation falls while the spread keeps widening up to $macron(rho) = +0.6$. Past that point the spread narrows slightly, yet the dispersion component does not decline, since sorting keeps moving inputs towards scalable firms, whose output responds most to a given markup by @eq:markup_response. The cost-weighted mean of $|epsilon_mu|$ rises from $17$ at the calibrated point to $70$ under perfect positive pairing, such that the same markups withhold more output. 


#v(0.5cm)
= Conclusion<sec:conclusion>
#v(0.2cm)
The modern literature on market power usually explains markups by productivity under common returns to scale. I introduce heterogeneity in RTS, measured from Orbis Spain using the #cite(<gandhi2020identification>, form: "prose") estimation method, to an #cite(<atkeson2008pricing>,form:"prose")-style Cournot oligopoly framework, and link it to productivity through a calibrated rank copula. Once firms differ in RTS, the same markup withholds very different amounts of output, so shares and markups alone cannot measure the cost of market power accurately.

Under the baseline calibrated $(alpha,nu)$-economy, I find that the cost of markups $lambda$, namely the welfare gap between the market and a planner that removes every markup, equals $32$ percent, more than double that of the $nu$-economy, which matches the same markup level and nearly the same sales tails with common returns to scale. I find that each dimension of heterogeneity contributes little on its own, with the joint interaction of the two accounting for almost half of the gap, such that productivity and scalability act as complements. The reason is that the two margins play different roles, as productivity drives the large shares and thus the high markups, while scalability determines how much output a given markup withholds, such that markups are most costly at firms that are both productive and scalable.

To investigate the components of this welfare gap, I split it in two separate ways. The first separates the dispersion of markups across firms, which accounts for a substantial share of the gap, from their common aggregate level, which accounts for the larger remainder. The second separates the misallocation of the inputs that the market already uses, which plays only a minor role, from the underuse of materials and capital, which accounts for most of the gap, with capital alone being its largest component. That is, markups in this framework lower welfare mostly by holding back the expansion of scalable firms.

Moving on to sorting, varying only the pairing of scalability and productivity, while holding both marginal distributions fixed, moves $lambda$ across a range far wider than that spanned by any dimension of heterogeneity in @tab:welfare_hetero, with the calibrated economy in the lower part of it. Sorting also creates market power, as aligning scalability with productivity raises the aggregate markup. Under perfect negative pairing, markups sit close to the CES floor, such that the whole welfare gap comes from their common level in the first split and from the underuse of materials and capital in the second. As the pairing turns positive, the dispersion component grows to about half of the total and misallocation to about a quarter, while capital remains the largest component throughout. Past random assignment, however, the correlation between markups and scalability falls, such that the welfare gap keeps rising due to a wider spread of markups and the reallocation of inputs toward scalable firms, whose output responds most to a given markup.

Some potential caveats qualify these conclusions, however, the first being that the comparison is between steady states, with labor and the number of firms held fixed, such that the transitional costs of reaching the planner's allocation are excluded. Second, scalability and its pairing with productivity are taken as given, while in practice they likely stem from investment in standardization or intangibles #cite(<argente2025scalable>) #cite(<hubmer2025scalable>) #cite(<deridder2024market>), and endogenizing them would be the most natural extension. Finally, the evidence comes from Spain alone, and the results remain sensitive to the choice of output anchor. Nonetheless, the results suggest that the welfare cost of markups depends on the expansion technology of the firms that charge them, such that the same markups are far more costly in an economy where the most productive firms are also the most scalable.

#colbreak()
// References only: smaller text and tighter entry spacing so they fit on one page.
#[
  #set text(size: 10.0pt)
  #set par(spacing: 0.6em)
  #bibliography("literature.bib", title: "References", style: "apa")
]

#colbreak()

// Appendix
#set figure(placement: none)
#counter(heading).update(0)
#set heading(numbering: "A.1.", supplement: [Appendix])
// Appendix figures and tables are numbered within their appendix (A1, B1, ...),
// restarting at each top-level appendix heading.
#set figure(numbering: n => numbering("A", counter(heading).get().first()) + str(n))
#show heading.where(level: 1): it => {
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
  it
}


= Derivations
#v(0.8cm)

== Demand, Markup and Cost Derivations
#v(0.2cm)

Conditional on market-level expenditure, the CES block in @eq:market_ces implies the demand system in @eq:firm_demand and the market-share expression in @eq:share_def. Under Cournot competition, firm $i$ internalizes the effect of its own quantity on the market price index and hence on its residual demand elasticity. The perceived elasticity can be written as
#math.equation(block:true, $epsilon_(j i,t)^(-1) = frac(1-s_(j i,t), gamma) + frac(s_(j i,t), eta),$)<eq:perceived_elasticity>
so that
#math.equation(block:true, $mu_(j i,t) = frac(epsilon_(j i,t), epsilon_(j i,t)-1).$)<eq:markup_eps>
Substituting @eq:perceived_elasticity into @eq:markup_eps yields the inverse-markup expression @eq:emx_markup reported in the text. This inverse-markup system is the object that must be solved before aggregation and moment construction.

As a pooled diagnostic, cost-weighting firm revenue-cost wedges across all producing firms yields the aggregate cost-weighted markup
#math.equation(block:true, $mu_("cw")^("model") = integral_0^1 sum_(i in A_j) lambda^c_(j i) mu_(j i) d j,$)<eq:mu_cw_identity>
where $lambda^c_(j i)$ is firm $i$'s share of aggregate variable cost. This pooled cost-weighted markup is the model analogue of the aggregate markup target $mu_("cw")^("data")$, which is itself constructed as a cost-weighted mean of firm-level price-marginal-cost ratios. It should not be confused with the accounting ratio of aggregate revenue to aggregate variable cost: because @eq:total_cost and @eq:pricing imply $T C_(j i) = alpha_(j i) p_(j i) y_(j i) slash mu_(j i)$, that ratio is instead
#math.equation(block:true, numbering: none, $mu^alpha_("cw") = integral_0^1 sum_(i in A_j) lambda^c_(j i) frac(mu_(j i), alpha_(j i)) d j,$)<eq:mu_cw_alpha>
which exceeds $mu_("cw")^("model")$ whenever $alpha_(j i) < 1$, since it also loads on the variable profit that decreasing returns generate even at a unit markup. The two coincide only under constant returns; at the calibrated point they are $1.182$ and $1.402$. Only $mu_("cw")^("model")$ is targeted.

The cost block follows from two nested Cobb-Douglas cost minimizations, first over capital and labor inside value added, which yields the input ratio in @eq:input_demand and the unit cost @eq:omega, and then over value added and materials, which yields @eq:omega_gross and, through @eq:production, the cost functions @eq:total_cost and @eq:marginal_cost.

#v(0.5cm)
== Planner Derivations<sec:planner_appendix>
#v(0.2cm)

Let the planner's Lagrangian attach multipliers $lambda_t^L$ and $lambda_t^K$ to the aggregate labor and capital constraints and let $psi_(j i,t)$ denote the marginal contribution of one more unit of firm output to the gross composite bundle. The planner's firm-level FOCs are the conditions reported in @eq:planner_labor_foc, @eq:planner_capital_foc, and @eq:planner_materials_foc. Together with @eq:production, they imply
#math.equation(block:true, numbering: none, $U_C(C_t,L_t) psi_(j i,t) alpha_(j i) phi.alt_v (1-a) frac(y_(j i,t), l_(j i,t)) = lambda_t^L,$)
#math.equation(block:true, numbering: none, $U_C(C_t,L_t) psi_(j i,t) alpha_(j i) phi.alt_v a frac(y_(j i,t), k_(j i,t)) = lambda_t^K,$)
#math.equation(block:true, numbering: none, $psi_(j i,t) (1-phi.alt_v) alpha_(j i) frac(y_(j i,t), m_(j i,t)) = 1.$)
The planner keeps the same exogenous sector firm set $A_j$ as the decentralized allocation. Hence the planner derivation needs no additional firm-selection condition: all differences between the market and planner allocations come from markup pricing and the induced allocation of labor, capital, materials, and output.

= Additional Figures and Tables <sec:aditional_figs>
#v(0.2cm)

#figure(
  block(width: 100%)[
    #align(center, image("figures/orbis/fig_sorting_curve_orbis_zipf.pdf", width: 100%))
    #align(left, text(size: 7.5pt)[_Note._ Each point is a solved economy at a copula strength $macron(rho)$, plotted at its realized corr$(alpha, nu)$ over active firms, and the dashed lines mark the calibrated economy. Panel A reports the total steady-state consumption-equivalent welfare gain $lambda$ from the market to the planner allocation, as in @tab:sorting. Panel B reports steady-state lifetime welfare as defined in @sec:planner_welfare, relative to the calibrated market, namely $W^r(macron(rho)) - W^(M E)(macron(rho)_"cal")$ for $r in {M E, P E}$. With labor and $chi$ fixed, this equals $ln[C^r(macron(rho)) / C^(M E)(macron(rho)_"cal")] / (1-beta)$. Transitional dynamics are excluded. Both panels use every copula sweep point except $macron(rho) = -0.15$, whose realized corr$(alpha, nu) = -0.13$ is indistinguishable from that of the calibrated economy. Segments between points are interpolations.])
    #v(0.5em)
  ],
  caption: [Consumption-equivalent gains and lifetime welfare against corr$(alpha, nu)$ along the copula sweep.],
)<fig:sorting_curve>


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
      [$a$], [common capital share inside value added], [$0.3333$, set to one third],
      [$eta$], [across-market elasticity (common)], [$1.2$, fixed from the literature],
      [$hat(y)$], [common anchor output (units normalization)], [$equiv 1$],
      table.hline(stroke: 0.7pt),

      table.cell(colspan: 3, align: left)[_B. External empirical inputs_],
      [$F^alpha$], [pooled scalability distribution], [firm-level gross-output returns to scale],
      [$phi.alt_v$], [value-added weight in gross output], [$0.517$, estimated from the GNR elasticities],
      [$omega_j$], [sector exposure weights], [normalized to uniform in the current baseline],
      table.hline(stroke: 0.7pt),

      table.cell(colspan: 3, align: left)[_C. Parameters determined inside the model_],
      [$xi$], [Pareto tail index of productivity $nu$], [$9.50$, targets sales tail indices $zeta$ (top 5% and top 2%)],
      [$N$], [Poisson mean firms per sector], [$604.0$, targets sales tail indices $zeta$ (top 5% and top 2%)],
      [$gamma$], [within-market elasticity (common)], [$7.88$, targets cost-weighted markup $mu_("cw")^("data")$],
      [$macron(rho)$], [rank-copula strength of $alpha$--$nu$], [$-0.166$, targets the top-versus-bottom RTS gap],
      [$underline(nu)$], [productivity scale / Pareto lower bound], [$8844.41$, solved by $tilde(y) = hat(y)$, not moment-targeted],
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ The four calibrated parameters $(xi, N, gamma, macron(rho))$ are solved jointly, with $(xi, N)$ disciplining the sales tail indices, $gamma$ pinning the markup level and $macron(rho)$ targeting the returns-to-scale gap. The productivity scale $underline(nu)$ is solved by the baseline anchoring condition $tilde(y) = hat(y) equiv 1$ rather than calibrated. External empirical inputs include $F^alpha$, $phi.alt_v$ and $omega_j$.])
    #v(0.5em)
  ],
  caption: [Parameterization used in the pooled sector-market steady-state implementation.],
)<tab:param_ss>




#figure(
  block(width: 100%)[
    #set text(size: 9pt)
    #show table: set block(below: 0.4em)
    #table(
      columns: (auto, 1fr, 1fr, 1fr, 1fr),
      column-gutter: 0.22cm,
      row-gutter: 0.12cm,
      align: (left, right, right, right, right),
      inset: (x: 0pt, y: 2.5pt),
      stroke: none,

      table.hline(stroke: 0.7pt),
      [], [Homogeneous], [$nu$-Economy], [$alpha$-Economy], [$(alpha,nu)$-Economy],
      table.hline(stroke: 0.4pt),
      table.cell(colspan: 5)[_Parameters_],
      [$gamma$ (within-market elast.)], [6.556], [7.776], [6.556], [7.875],
      [$xi$ (Pareto tail of $nu$)], [$infinity$ (fixed)], [5.05], [$infinity$ (fixed)], [9.50],
      [$N$ (Poisson mean firms)], [604.0 (fixed)], [1997.9], [604.0 (fixed)], [604.0],
      [$macron(rho)$ (rank copula $alpha$--$nu$)], [none], [none], [none], [$-0.166$],
      [$alpha$ (scalability)], [0.957 (common)], [0.954 (common)], [$F^alpha$], [$F^alpha$],
      [$eta$ (across-market elast.)], [1.2 (fixed)], [1.2 (fixed)], [1.2 (fixed)], [1.2 (fixed)],
      table.hline(stroke: 0.4pt),
      table.cell(colspan: 5)[_Moments (data)_],
      [$mu_("cw")$ (1.1815)], [1.1815], [1.1817], [1.1815], [1.1816],
      [Sales tail $zeta$, top 5% (1.1079)], [55.03\*], [1.0962], [11.96\*], [1.1078],
      [Sales tail $zeta$, top 2% (1.1451)], [54.55\*], [1.1584], [9.58\*], [1.1452],
      [RTS gap, top 5% $-$ bottom 50% (0.0460)], [0\*], [0\*], [$-0.030$\*], [0.0460],
      table.hline(stroke: 0.7pt),
    )
    #align(left, text(size: 7.5pt)[_Note._ Each economy is calibrated on its free parameters to the targets it can reach; \* marks moments it does not target. A common $nu$ is $xi = infinity$, and a common $alpha$ is the harmonic mean of $F^alpha$ below the cap $0.99 gamma slash (gamma - 1)$. With common $nu$ the economy generates no Pareto tail in sales, so no target identifies $N$, which stays at its calibrated value; $macron(rho)$ is undefined once either dimension is common, and $eta$ is fixed at 1.2 as in @tab:calib_fit. $N$ in the $nu$-economy ends just below its upper bound of 2000.])
    #v(0.5em)
  ],
  caption: [Calibration of the four economies of @tab:welfare_hetero.],
)<tab:recal_calib>
