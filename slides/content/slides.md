---
bibliography: "bibliography.bib"
link-citations: true
urlcolor: "blue"
---

### Total Variation in theory

$$ 𝖳𝖵(\mathbb{P}_θ,\mathbb{P}_{θ\prime}) = \max_{A \subset E}|\mathbb{P}_θ(A) - \mathbb{P}_{θ\prime}(A)| $$

### Total Variation of continuous probability distributions

$$ 𝖳𝖵(\mathbb{P}_θ,\mathbb{P}_{θ\prime}) = \frac{1}{2} \int_E|f_θ(x) - f_{θ\prime}(x)|\,dx $$

### Kullback-Leibler divergence aka. relative entropy

$$ 𝖪𝖫(\mathbb{P}_θ,\mathbb{P}_{θ\prime}) =  \int_Ef_θ(x)\log\left(\frac{f_θ(x)}{f_{θ\prime}(x)}\right)\,dx $$

- not symmetric (compare to variants of the Jensen–Shannon divergence)
- greater or equal to zero
- definite, i.e. one minimum with $𝖪𝖫(\mathbb{P}_θ,\mathbb{P}_{θ\prime}) = 0$
- not a distance
- a divergence
- can be statistically estimated (by the law of large numbers via an average)
