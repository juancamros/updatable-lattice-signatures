# Changelog

## [v0.2.0] - 2026-09-13

### USGPV — Reduction of the update-token spectral norm

The update-token generation procedure has been modified with the objective of
reducing the spectral norm of the token while preserving the required algebraic
relation

$$ A_{e+1}\Delta_{e+1} = A_e \pmod q $$

Since the update token is generated only once per epoch and is kept private, its
statistical distribution does not need to match the spherical distribution required
for signatures. This allows the covariance matrix used during token generation to be
adapted in order to reduce the spectral norm of the resulting token, while preserving
the correctness and sampling conditions required by the scheme.

The following modifications have been introduced:

- The gadget-lattice basis $S$ has been reordered in order to reduce the norm of
  its Gram-Schmidt orthogonalization $\bar S$.

- The Gaussian scaling parameter $r$ has been reduced accordingly, taking into
  account the reduced Gram-Schmidt norm and the value of $\sigma_G$.

- Signature generation and update-token generation now use different covariance
  matrices.

- Fresh signatures generated at epoch $e$ use the epoch-dependent spherical
  covariance associated with $s_e$. Updated signatures are corrected with
  additional Gaussian noise so that they target the same epoch-dependent dispersion
  as freshly generated signatures.

- Update tokens use a dedicated covariance matrix defined from

  $$ M_{\mathrm{boundary}} = B(2I_w+\Sigma_G)B^T $$

  and

  $$ \Sigma_{\mathrm{token}} = (1+\delta)M_{\mathrm{boundary}} + \eta I_m $$

  with

  $$ \delta = 0.02, \qquad \eta = \sigma_G $$

  The factor $\delta=0.02$ introduces a small safety margin above the covariance
  boundary, while the additive term $\sigma_G I_m$ provides a minimum isotropic
  margin.

- The covariance used for the perturbation during token generation is therefore

  $$ \Sigma_p = \Sigma_{\mathrm{token}} - B\Sigma_GB^T $$

#### Experimental evaluation of the update-token norm

The modified update-token sampler was evaluated over 100 independently generated
tokens using the current parameter set.

The spectral norm of the generated tokens showed the following distribution:

- Mean: $2908.61$
- Median: $2902.37$
- Standard deviation: $41.20$
- Minimum: $2813.26$
- Maximum: $3016.12$

Based on these results, the fixed update-growth parameter was reduced from

$$ c_{\mathrm{upd}} = 22900 $$

to

$$ c_{\mathrm{upd}} = 3200 $$

This corresponds to an approximately $86.0\%$ reduction in
$c_{\mathrm{upd}}$.

The selected value $c_{\mathrm{upd}}=3200$ leaves a margin of approximately
$6.1\%$ above the largest token spectral norm observed in the 100-sample
experiment.

The value $c_{\mathrm{upd}}=3200$ is therefore an empirically selected parameter
based on the observed token-norm distribution. It should not be interpreted as a
proven theoretical upper bound.

#### Effect on epoch-dependent parameter growth

Reducing $c_{\mathrm{upd}}$ directly slows the growth of the epoch-dependent
Gaussian parameters, since

$$ s_{e+1} = c_{\mathrm{upd}}(s_e+1) $$

The reduction of the update-token norm therefore affects not only the token itself,
but also the evolution of the complete scheme through successive epochs.

A scalability experiment was performed using the same dimension, initial
parameters and random seed, changing the fixed update-growth parameter from

$$ c_{\mathrm{upd}}=22900 $$

to

$$ c_{\mathrm{upd}}=3200 $$

With the previous value $c_{\mathrm{upd}}=22900$, the Gaussian parameters evolve
as

$$ s_1 \approx 5.36\times10^5 $$

$$ s_2 \approx 1.23\times10^{10} $$

$$ s_3 \approx 2.81\times10^{14} $$

and

$$ s_4 \approx 6.44\times10^{18} $$

At epoch 4, the corresponding Gaussian scale reaches

$$ r s_4 \approx 3.85\times10^{19} $$

which is approximately $4.18$ times larger than the maximum value representable
by `np.int64`. Therefore, under the current numerical implementation, epochs up to
$e=3$ can be considered numerically reliable, while epoch $e=4$ already exceeds
the available integer range.

With the new value $c_{\mathrm{upd}}=3200$, the parameter growth becomes

$$ s_1 \approx 7.49\times10^4 $$

$$ s_2 \approx 2.40\times10^8 $$

$$ s_3 \approx 7.67\times10^{11} $$

and

$$ s_4 \approx 2.45\times10^{15} $$

At epoch 4

$$ r s_4 \approx 1.47\times10^{16} $$

which remains well inside the `int64` range. The numerical limit is instead reached
at epoch 5, where

$$ r s_5 \approx 4.70\times10^{19} $$

Under the current `int64` implementation, the experimentally reliable update depth
therefore increases from three to four successive updates.

At epoch 4, the Gaussian parameter is reduced from approximately

$$ 6.44\times10^{18} $$

to

$$ 2.45\times10^{15} $$

corresponding to a reduction by a factor of approximately

$$ 2623 $$

The corresponding signature bound is also substantially reduced, from approximately

$$ \beta_4 \approx 1.18\times10^{21} $$

to

$$ \beta_4 \approx 4.49\times10^{17} $$

The update-token spectral norms remain essentially unchanged between both
experiments. For example, the first four transition tokens have norms approximately

$$ 2943,\quad 2903,\quad 2848,\quad 2893 $$

in both configurations. This experimentally confirms that the dedicated token
sampler is decoupled from the epoch-dependent signature dispersion.

Despite this improvement, the current construction is still not scalable to a large
number of epochs. Both $s_e$ and the signature bounds $\beta_e$ continue to grow
multiplicatively after successive updates.

Therefore, reducing the update-token norm and $c_{\mathrm{upd}}$ significantly
slows parameter growth and extends the practical update depth of the implementation,
but does not eliminate the underlying scalability limitation.


### CRSST21

The CRSST21 implementation has been revised in order to follow the same general
implementation framework used in USGPV.

The common components of both implementations have been aligned, including:

- epoch management;
- trapdoor generation;
- gadget construction;
- preimage sampling;
- Gaussian parameter management;
- update-token generation;
- signature bounds;
- algebraic validation of the update token.

This allows CRSST21 and USGPV to be evaluated under a common computational
framework and makes their update mechanisms directly comparable.

The fundamental distinction between both schemes is preserved in the signature
update procedure.

CRSST21 performs a deterministic update:

$$ u_{e+1} = \Delta_{e+1}u_e $$

USGPV instead introduces an additional Gaussian correction term:

$$ u_{e+1} = \Delta_{e+1}u_e+r_e $$

The correction noise in USGPV is sampled so that an updated signature targets the
same epoch-dependent Gaussian dispersion as a freshly generated signature at the
same epoch.

CRSST21 retains its deterministic update behavior and does not introduce this
additional Gaussian correction term.


### Tests

A `pytest` test suite has been added for both USGPV and CRSST21.

The tests cover the main algebraic, numerical and implementation invariants of both
schemes.

The tested properties include:

- trapdoor correctness:

  $$ AB = G \pmod q; $$

- gadget-kernel correctness:

  $$ GS = 0 \pmod q; $$

- correctness of the binary decomposition and gadget preimage sampler;

- correctness of `SamplePre`;

- validation of epoch-dependent Gaussian parameters and signature bounds;

- dedicated covariance for update-token generation;

- explicit use of

  `SamplePre(..., mode="token")`

  during update-token generation;

- update-token correctness:

  $$ A_{e+1}\Delta_{e+1} = A_e \pmod q; $$

- complete

  `Sign -> Next -> Update -> Verify`

  integration flows;

- preservation of the signed-message relation after an update;

- deterministic signature updates in CRSST21;

- Gaussian-corrected signature updates in USGPV;

- correct use of the epoch-dependent signing dispersion in USGPV;

- Gram-Schmidt rank-deficiency detection;

- preservation of the internal Gaussian covariance state;

- detection and documentation of the current numerical limitations associated with
  `np.int64` when the epoch-dependent Gaussian parameters become too large.

Integration tests are explicitly marked using the `pytest` `integration` marker.
