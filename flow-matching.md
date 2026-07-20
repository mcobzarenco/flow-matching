# Flow Matching — working notes & agent context

Context document for continuing a deep technical thread on flow matching. Prior session covered: pushforward measures and why normalizing flows require invertibility, the statistical meaning of Jacobian rank loss, the full CFM theorem with proofs, the continuity equation from first principles, velocity–score duality, and an incrementally-built PyTorch MNIST implementation (current state in §3). Long-horizon motivation: the action expert for **Thor** (custom humanoid, SmolVLA/π0-style VLA), eventually Riemannian FM for SO(3)/S¹ joints.

---

## 1. Theory established (results only; proofs were worked in full)

### 1.1 Pushforward & change of variables

- $x\sim p$, $y=f(x)$ ⟹ $y \sim f_\#p$, $(f_\#p)(A)=p(f^{-1}(A))$. Sampling never needs invertibility; **density evaluation does**.
- Diffeomorphism: $p_y(y) = p_x(f^{-1}(y))\,|\det J_f(f^{-1}(y))|^{-1}$. Non-injective full-rank: sum over preimages $p_y(y)=\sum_{x_i\in f^{-1}(y)} p_x(x_i)/|\det J_f(x_i)|$.
- CNF instantaneous change of variables: $\frac{d}{dt}\log p_t(x_t) = -\nabla\!\cdot v_t(x_t)$ along trajectories.

### 1.2 Why NF literature imposes invertibility

Three cascading computational facts:
1. **Rank-deficient $J$ on positive-mass sets** ⟹ pushforward singular w.r.t. Lebesgue ⟹ $\mathrm{KL}(p_{\text{data}}\|f_\#p)=\infty$ identically in $\theta$: MLE objective flat, no gradient. Unconstrained nets deliver this (ReLU cells with singular affine maps, dead regions → atoms, width bottlenecks → global rank cap).
2. Full-rank but many-to-one ⟹ preimage sum intractable (exponentially many ReLU regions; exact inversion NP-hard). Folding (volume preserved, density survives but intractable) vs collapsing (rank loss, density destroyed) — distinct failure modes; entangled in 1D, separable in $n\ge2$ (e.g. $e^z$ folds without collapsing).
3. Even bijective: $\det J$ is $O(d^3)$ ⟹ the architecture zoo (coupling layers, autoregressive, i-ResNets via Banach fixed point).

Each way of paying the tax = a model family: GAN (keep sampling, drop density, adversarial divergence), VAE ($y=f(x)+\varepsilon$ resurrects density, ELBO; encoder = learned stochastic right-inverse), SurVAE (systematic surjective layers), diffusion (ELBO stack; probability-flow ODE sampler is a CNF again).

Costs *of* bijectivity: diffeomorphisms preserve topology (Gaussian → disconnected support impossible exactly; Cornish et al.: bi-Lipschitz constant must diverge — the numerical non-invertibility seen in Glow); dimension preservation vs manifold hypothesis (likelihood blow-up, $\log|\det J|\to\infty$).

**FM resolution:** parameterize the *generator* $v_\theta$ (any architecture), not the transport map. Picard–Lindelöf makes the flow map a diffeomorphism for free (Lipschitz field ⟹ trajectories can't cross ⟹ invertible, inverse = integrate backwards). Determinant relaxes to a divergence, and the CFM objective never computes even that.

### 1.3 Jacobian rank loss, statistically

- Local linearization: $\mathcal N(x_0,\epsilon^2 I) \mapsto \approx \mathcal N(f(x_0), \epsilon^2 JJ^\top)$; $\mathrm{rank}(J)=r<n$ ⟹ degenerate covariance. Rank deficiency = local nonlinear version of singular $\Sigma$.
- SVD: null singular directions annihilated; output satisfies $n-r$ exact local functional constraints ⟹ no joint density. Minimal example $Y=(X,X)$.
- Benign vs fatal: isolated rank loss OK ($x^3$: integrable density spike; Sard: critical values null). Fatal requires rank deficiency on positive-$p$-mass sets.
- Generic pushforward Lebesgue-decomposes: $\mu_{\mathrm{ac}} + \mu_{\mathrm{sing}} + \sum a_i\delta_{y_i}$.
- Estimation consequence: data a.s. off model support ⟹ $\log$-lik $=-\infty$ ∀θ, flat objective. Near-singular continuum: $\log|\det J|=\sum\log\sigma_i$ dominated by $\sigma_{\min}$; MLE on manifold data drives $\sigma_{\min}\to0$. Same geometry as Arjovsky's GAN analysis (mutually singular ⟹ JS saturates ⟹ WGAN); noise convolution = VAE/denoising fix.

### 1.4 CFM theorem (both halves proved)

Setup: $z=(x_0,x_1)\sim\pi\in\Pi(p_0,q)$, bridge $x_t=(1-t)x_0+tx_1$, conditional field $u_t(x|z)=x_1-x_0$.

**(a) Marginalization.** Continuity equation is **linear in (density, flux)**. Fluxes superpose ($j_t=\int u_t(\cdot|z)p_t(\cdot|z)\pi(z)dz$), velocities don't; the marginal field is the quotient
$$u_t(x) = \frac{j_t(x)}{p_t(x)} = \mathbb E[\,x_1-x_0 \mid x_t=x\,]$$
with Bayes posterior weights $\pi(z|x_t{=}x)$. Proof that this field *generates* $\{p_t\}$ (i.e. $(\phi_t)_\#p_0 = p_t$ ∀t):
(i) any flow's density film solves the CE (weak form, test functions);
(ii) the stochastic bridge's one-time marginals solve the *same* CE — differentiate $\mathbb E\varphi(x_t)$ along the bridge, condition on $x_t$ (tower property);
(iii) uniqueness for the linear PDE via duality: solve backward transport $\partial_t\varphi + u\cdot\nabla\varphi=0$ along characteristics, conclude $\rho_t\equiv p_t$.
Key conceptual point: the CE sees only (density, flux), never trajectories — bridge process (crossing, non-Markov) and ODE flow (non-crossing) are different "actors" producing the same "film". Regularity caveats: need $p_t>0$ where dividing (true for $t<1$: $p_t(\cdot|x_1)=\mathcal N(tx_1,(1-t)^2I)$); Lipschitzness can degrade as $t\to1$ on manifold data.

**(b) Regression.** $\arg\min_v \mathbb E\|v(X)-Y\|^2 = \mathbb E[Y|X]$. With $X=(x_t,t)$, $Y=x_1-x_0$, population minimizer is exactly $u_t$. Stronger:
$$\mathcal L_{\mathrm{CFM}}(\theta) = \mathcal L_{\mathrm{FM}}(\theta) + \mathbb E_{t,x_t}\,\mathrm{Var}[x_1-x_0\mid x_t]$$
— identical gradients ∀θ (cross terms match by tower property). Structurally = denoising score matching. **Practical corollary: training loss plateaus at the variance floor; judge convergence by samples, never the loss curve.**

**Coupling freedom.** Boundary conditions touch only marginals ⟹ every $\pi\in\Pi(p_0,q)$ valid; independence is the samplable one (same move as diffusion's $\varepsilon\perp x_1$). Coupling determines the *interior* film: crossings of conditional straight lines are laundered into **curvature** of the marginal trajectories (averaging at crossing points; Picard–Lindelöf forbids marginal crossings). Curvature = sampling cost. Minibatch-OT coupling ⟹ fewer crossings ⟹ smaller posterior variance ⟹ straighter paths.

Worked example (1D, $p_0=q=\mathcal N(0,1)$, independent): $u_t(x) = \frac{2t-1}{t^2+(1-t)^2}x$; field freezes at $t=\frac12$ (opposing currents cancel), $\phi_t(x_0)=x_0\sqrt{t^2+(1-t)^2}$, endpoint map = identity, positive kinetic energy for null transport. OT coupling: $u\equiv0$.

**Non-uniqueness footnote:** given $\{p_t\}$, generating fields differ by any $w_t$ with $\nabla\!\cdot(p_t w_t)=0$. The minimal-kinetic-energy representative is a gradient field (Benamou–Brenier / Otto geometry); the FM posterior-mean field is generally a different representative.

### 1.5 Continuity equation

$$\partial_t p_t + \nabla\!\cdot(p_t u_t) = 0, \qquad j_t = p_t u_t \text{ (flux)}$$
Derivation: conservation axiom (mass changes only by boundary crossing — postulate, valid for continuous trajectories, false for jump processes) → swept-cylinder argument gives per-patch crossing rate $p_t(u_t\cdot n)\,dS$ (only the normal component crosses) → divergence theorem converts $\oint_{\partial V}$ to $\int_V$ → arbitrariness of $V$ localizes to the pointwise PDE. Weak form (test functions $\varphi$): $\frac{d}{dt}\int\varphi\,p_t = \int\nabla\varphi\cdot u_t\,p_t$ — survives Diracs. Material derivative: particle acceleration $= \partial_t u + (u\cdot\nabla)u$; **straight marginal trajectories ⟺ zero material derivative** (what OT coupling buys; enables few-step sampling).

### 1.6 Velocity–score duality

- $u_t \ne \nabla p_t$ (dimensions; CE underdetermines the field pointwise). $u_t$ is Lagrangian: $\dot x_t$ of the ODE trajectories. Conditional $\dot x_t = x_1-x_0$ (the training target) vs marginal $u_t = \mathbb E[\dot x_t|x_t]$ — velocities of two *different* path families; CFM says averaging the first yields the second.
- Heat equation is a CE with $u=-\nabla\log p$: diffusion = deterministic transport down the score.
- Linear path Tweedie: $u_t(x) = \frac{x + (1-t)\nabla\log p_t(x)}{t}$ — velocity and score affinely interconvertible with $t$-only coefficients. FM and diffusion = same object, different parameterization. This affinity is what makes CFG valid on velocity fields.

### 1.7 Training/sampling improvements (theory-linked)

- **EMA** (β=0.999, `torch.optim.swa_utils.AveragedModel`): SGD orbits the basin; trajectory average sits deeper. Highest quality-per-line change; sample from the shadow copy.
- **CFG**: train with 10% label→∅ dropout; sample $\tilde u = (1+w)u(\cdot|y) - w\,u(\cdot|\varnothing)$. Samples tilted density $\propto p(x|y)[p(x|y)/p(x)]^w$; via Tweedie the added vector is $\propto \nabla\log p_t(y|x)$. Suppresses inter-mode mass (digit morphs). *Currently removed from the code by choice.*
- **Logit-normal $t$** (`sigmoid(randn())`, SD3): concentrates training mid-path where posterior variance (target difficulty) peaks; endpoints nearly trivial.
- **Heun**: second-order; 50 Heun steps > 100 Euler on curved fields. Final step kept plain Euler: from $t=1{-}dt$ it computes $x+(1{-}t)u_t(x) = \mathbb E[x_1|x_t]$ — the posterior-mean jump.
- **Conditional > unconditional** (last topic discussed). Law of total variance: conditioning turns the between-class variance term from label noise into signal ⟹ lower floor, cleaner trunk. First-order recipe: **ancestral marginalization** — sample $y\sim p(y)$, run the conditional flow; exact by CE linearity (extend $z$ to $(x_0,x_1,y)$), and avoids making the ODE learn a classifier-sharp separatrix (topology mismatch → morphs → the smoothed boundary is where hybrids are born). Deletes between-class crossings ⟹ straighter paths. The ∅-token branch itself is second-order (undertrained at 10% dropout). Empirical anchor: conditional ≫ unconditional ImageNet diffusion; RCG manufactures conditioning when labels absent. Robotics analogue: condition the action expert on VLM features to shrink the posterior the flow averages over.
- **Architectural ceiling**: MLP treats 784 pixels as unordered coordinates — spatially uncorrelated output errors = irreducible residual grain. Small U-Net or DiT (patch tokens + adaLN conditioning) dominates all training tricks combined. Same lesson for Thor: leverage is in matching architecture to output-space structure (action-chunk tokenization/conditioning), not trunk width.

---

## 2. Model/code conventions in use

- Pre-LN residual blocks; **GEGLU** FF branch $W_p(\mathrm{GELU}_{\tanh}(W_a x)\odot W_b x)$ — Gemma's MLP (SiLU gate would be SwiGLU/LLaMA/π0). Fused gate+value as one `Linear(h, 2g)` + `chunk`. Inner width $g=\frac23\,\mathrm{mult}\cdot h$ (Shazeer parameter-matching convention).
- **Zero-init** the last projection of every block ⟹ whole stack is identity at init; residual branches switch on gradually.
- Fourier time embedding: random `freqs ~ N(0,1)*10`, features $[\sin,\cos](2\pi t\,\text{freqs})$.
- Data in $[-1,1]$; independent coupling $x_0\sim\mathcal N(0,I)$; target $x_1-x_0$; MSE.
- Fixed eval noise `z0` drawn once ⟹ each grid cell tracks $\phi_1(z_0^{(i)})$ across epochs (deterministic flow map) — comparable frames.
- Hardware: ThinkPad P1 Gen 7, RTX Ada, Ubuntu 26.04, CUDA. User also builds cuTile Rust GPU kernels and Rust/Embassy firmware; comfortable with both stacks.

## 3. Current code (last working state — unconditional)

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision import datasets, transforms, utils
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- model ----------

class Block(nn.Module):                   # pre-LN residual GEGLU block
    def __init__(self, h, mult=2):
        super().__init__()
        g = max(int(2 * mult * h / 3), 1)     # inner width, 2/3 convention
        self.norm = nn.LayerNorm(h)
        self.gate_val = nn.Linear(h, 2 * g)   # fused gate+value projection
        self.proj = nn.Linear(g, h)
        nn.init.zeros_(self.proj.weight)      # block is identity at init
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        a, b = self.gate_val(self.norm(x)).chunk(2, dim=-1)
        return x + self.proj(F.gelu(a, approximate='tanh') * b)

class VF(nn.Module):                      # v_theta(x, t) — unconditional
    def __init__(self, d=784, h=1024, tdim=64, depth=6):
        super().__init__()
        self.register_buffer('freqs', torch.randn(tdim // 2) * 10)
        self.inp = nn.Linear(d + tdim, h)
        self.blocks = nn.Sequential(*[Block(h) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))
    def forward(self, x, t):
        a = 2 * torch.pi * t[:, None] * self.freqs[None]
        return self.out(self.blocks(self.inp(torch.cat([x, a.sin(), a.cos()], -1))))

# ---------- sampling ----------

@torch.no_grad()
def sample(x0, steps=50):                 # Heun, EMA weights; integrate from x0
    net = ema.module.eval()
    n, dt = x0.size(0), 1 / steps
    x = x0.clone()
    def v(x, t):
        return net(x, torch.full((n,), t, device=dev))
    for i in range(steps):
        v0 = v(x, i * dt)
        if i == steps - 1: x = x + dt * v0; break        # final step: mean jump
        x = x + dt * (v0 + v(x + dt * v0, (i + 1) * dt)) / 2
    return ((x.view(n, 1, 28, 28) + 1) / 2).clamp(0, 1)

def save_samples(path='samples.png', scale=4):
    img = utils.make_grid(sample(z0), nrow=8)
    img = resize(img, [img.shape[-2] * scale, img.shape[-1] * scale],
                 interpolation=InterpolationMode.NEAREST_EXACT)
    utils.save_image(img, path)

# ---------- training ----------

ds = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())
dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True,
                num_workers=2, pin_memory=True)

model = VF().to(dev)
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

z0 = torch.randn(64, 784, device=dev)     # fixed eval noise

for epoch in range(60):
    total, seen = 0.0, 0
    for x1, _ in dl:                      # labels discarded
        x1 = x1.to(dev).view(-1, 784) * 2 - 1
        x0 = torch.randn_like(x1)                        # independent coupling
        t = torch.sigmoid(torch.randn(x1.size(0), device=dev))  # logit-normal t
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1     # linear conditional path
        loss = (model(xt, t) - (x1 - x0)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update_parameters(model)
        total += loss.item() * x1.size(0); seen += x1.size(0)
    print(f'epoch {epoch:3d}  loss {total / seen:.4f}')
    save_samples()                                       # same name, auto-reloads
```

History of this file within the session: plain MLP → residual pre-LN blocks (zero-init) → GEGLU blocks → +class conditioning, CFG, EMA, logit-normal t, Heun → CFG machinery removed (kept conditioning) → conditioning removed entirely (current). Expect occasional between-mode digit morphs — the unconditional model is *correct* to produce them.

## 4. Agreed next steps

1. **Re-add class conditioning without CFG**: keep `nn.Embedding(10, tdim)`, no label dropout, no null token. For unconditional samples, marginalize ancestrally: `y = torch.randint(0, 10, ...)` (or fixed `arange(10).repeat_interleave(8)` for the eval grid), then run the conditional flow. Fixed `z0` + fixed `y` for comparable frames.
2. Optional squeeze: minibatch-OT coupling (`scipy.optimize.linear_sum_assignment` on pairwise $\|x_0-x_1\|^2$ per batch — weak signal at 784 raw pixels but principled), cosine LR decay, `depth=8`.
3. **Architecture jump** (the real ceiling): small U-Net, or DiT on 4×4 patches with adaLN conditioning — reuses the GEGLU blocks + attention.
4. Longer horizon: port the pattern to Thor's action expert — flow matching over action chunks conditioned on VLM features (π0/SmolVLA); then Riemannian FM for SO(3)/S¹ joints (geodesic conditional paths replace straight lines).

## 5. Interaction conventions

Terse, mathematically dense, full derivations over summaries, LaTeX with explicit tensor shapes, direct answers without hedging, no unexplained acronyms. Code: compact, single-block, comments only where they mark a design decision. Flag genuine uncertainty and open regularity caveats honestly (e.g. $t\to1$ Lipschitz degeneration, non-uniqueness of the generating field).
