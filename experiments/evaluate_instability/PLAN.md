# Motivation

In earlier experiments, we identified unstable data points by running more than 100 EqM sampling steps starting from each clean image and measuring the L2 drift between the original image and the final sampled result. While this revealed instability patterns, the approach has two major limitations:

1. **High computational cost:**
    
    Evaluating instability requires running 100+ EqM updates per datapoint, making it too expensive to compute frequently (e.g., during training).
    
2. **Poor locality:**
    
    Because the trajectory evolves for many steps, the measured drift is influenced not only by the manifold structure near the initial point, but also by the geometry of the surrounding regions. Thus, the instability score is not strictly a *local* property of the manifold.
    

Given these limitations, our goal is to develop an evaluation method that is:

- **computationally much cheaper**, ideally requiring only a few forward/backward passes; and
- **more directly tied to the local manifold structure** around each datapoint, rather than depending on long-range dynamics.

# Possible Approaches

## 1. Divergence via Hutchinson’s estimator (cheap, scalable)

### **Goal:** Estimate

[
\text{div}(f)(x) = \sum_i \frac{\partial f_i}{\partial x_i}
]

### **Trick:** Using a random vector ( v\sim \mathcal{N}(0,I) ),

[
\text{div}(f)(x) = \mathbb{E}_v [ v^\top J(x) v]
]

Compute using PyTorch:

```python
v = torch.randn_like(x)  # same shape as image
f_x = f(x)
dot = (f_x * v).sum()
div = torch.autograd.grad(dot, x, create_graph=False)[0].dot(v)
```

### **Cost:**

* 1 forward pass
* 1 backward pass

### **Interpretation:**

* High positive divergence → instability / expansion
* Negative divergence → locally contracting (stable)
* Close to 0 → neutrally stable → drift likely

### **Pros:**

* Very cheap
* Works for 256×256 images
* Gives a scalar instability score

**This is definitely the most practical method for your setting.**



## 2. Largest Jacobian eigenvalue via power iteration (approximate Lyapunov exponent)

You want:

[
\lambda_{\max}(x) = \max \text{eig}(J(x)).
]

### **Algorithm:** Power iteration using JVP/VJP.

Start with a random vector ( v_0 ).

Repeatedly compute:

[
v_{k+1} = \frac{J(x) v_k}{| J(x) v_k |}
]

Use vector-Jacobian product:

```python
v = torch.randn_like(x)
v = v / v.norm()

for _ in range(K):  # K = 5–10 iterations
    f_x = f(x)
    Jv = torch.autograd.grad(f_x, x, v, retain_graph=True)[0]
    v = Jv / (Jv.norm() + 1e-8)

# Estimate eigenvalue
lambda_max = (v * Jv).sum() / (v * v).sum()
```

### **Cost:**

* One backprop per iteration
* Typically **5–10 iterations** enough

This still works for DiT-XL/2 scale models.

### **Interpretation:**

* ( \lambda_{\max} > 0 ): unstable region
* ( \lambda_{\max} \approx 0): marginal stability → drifting
* ( \lambda_{\max} < 0 ): stable

### Why this matters for EqM

EqM *intends* to learn a gradient field.
But because it's learned implicitly, it may form:

* spirals
* sources
* saddles

even near data points.

This method diagnoses them precisely.



## 3. One-step instability / amplification metric (ultra-cheap)

Instead of computing eigenvalues explicitly, measure:

[
S(x) = \frac{| x - \eta f(x) - x |}{|x|}
= \eta \frac{|f(x)|}{|x|}
]

This is the **local magnification** of a single optimization step.

If multiple small steps cause the image to rapidly drift → unstable region.

### Pros:

* Requires only f(x)
* No gradients needed
* Extremely cheap

But it only measures "force magnitude", not curvature.



## 4. Empirical perturbation instability (cheap but approximate)

Perturb image slightly:

[
x' = x + \sigma \epsilon
]

Apply 10 steps of EqM sampling to x and x′:

[
x_{t+1} = x_t - \eta f(x_t)
]

Compute divergence:

[
\Delta = |x_{10} - x'_{10}|_2
]

If Δ grows → instability.

This correlates strongly with λ_max but is more expensive.


## 5. Symmetric Jacobian Error as a Local Instability Metric

EqM models are intended to learn **equilibrium dynamics**, where the predicted vector field
( f(x) \approx \nabla E(x) )
represents the gradient of an implicit energy function. A fundamental mathematical property of any gradient field is that its Jacobian must be **symmetric**:

[
J(x) = \nabla f(x) = \nabla^2 E(x), \qquad J(x) = J(x)^\top.
]

Thus, deviations from Jacobian symmetry directly measure how strongly the learned field violates the equilibrium (energy-based) assumption. In practice, such violations manifest as **rotational or curl components** in the vector field, which lead to **unstable or drifting trajectories** during EqM sampling.

To quantify this, we evaluate the **antisymmetric component** of the Jacobian using only Jacobian–vector products (no explicit Jacobian construction):

[
A(x)v = J(x)v - J(x)^\top v.
]

Both (Jv) and (J^\top v) can be computed efficiently for high-dimensional diffusion models via automatic differentiation. The magnitude of the antisymmetric component,

[
|A(x)v|,
]

captures the degree to which the dynamics around (x) deviate from a true gradient flow. Because the scale of (Jv) varies across datapoints, we use a **relative symmetry error**:

[
S_{\text{sym}}(x) =
\frac{|J(x)v - J(x)^\top v|}
{|J(x)v|},
]

with (v \sim \mathcal{N}(0, I)). This normalization makes the metric **comparable across datapoints**, providing a local, scale-invariant measure of instability.

Low symmetry error ((S_{\text{sym}} \approx 0)) indicates that the vector field behaves like a conservative gradient, suggesting a stable local energy basin. High symmetry error indicates that the model has learned significant non-equilibrium structure (curl), which often correlates with **sampling drift, manifold instability, and failure of EqM’s theoretical assumptions** near the datapoint.

This metric is computationally lightweight—requiring one forward evaluation and two backward passes—and scales to large models such as DiT or U-Net variants. It serves as a principled diagnostic for assessing **local manifold stability** and **energy-landscape consistency** in EqM.

## 6. Multi-step backward consistency (cycle error)

Given x:

1. Apply one forward step:
   [
   y = x - \eta f(x)
   ]

2. Then a backward step:
   [
   x' = y + \eta f(y)
   ]

Full consistency means (x' \approx x).

Instability → the field is not locally consistent:

[
C_{\text{cycle}} = |x' - x|
]

## 7. Local flow dispersion

Choose N tiny perturbations around x:

Run only **5–10 steps** (short horizon):

Compute the volume of the convex hull of the resulting points.

Large expansion → unstable.

This GT captures whether the local vector field is divergent.


# Defining ground-truth (GT) instability

## 1. Long-horizon drift

[
D_{\text{drift}}(x)=|x_K - x_0|_2
]
Pros: direct interpretation.
Cons: global, expensive, non-local.



## 2. Long-horizon divergence from nearby perturbations

Instead of measuring drift from x alone, measure *separation between two perturbed versions*:

Pick two close initial points:
[
x_0,\quad x_0' = x_0 + \epsilon
]

Run EqM sampling for K steps:

[
d_K = |x_K - x_K'|
]

Interpretation:

* If ( d_K ) grows → **chaotic / unstable region**
* If ( d_K ) shrinks → **stable attractor**

This is a discrete approximation of **Lyapunov instability**.



## 3. Multi-step amplification of random noise directions

Generate N random directions (v_i):

[
x_i = x_0 + \delta v_i
]

Run sampling for K steps and compute:

[
D_i = |x_{i,K} - x_K|
]

GT instability could be:

* **max amplification**
* **average amplification**
* **variance of amplification**

This captures sensitivity to perturbations in arbitrary directions.



## 4. Trajectory curvature / bending

During sampling, compute:

[
C = \sum_{t} \frac{|x_{t+1} - 2x_t + x_{t-1}|}{|x_{t+1} - x_t|}
]

High curvature → unstable or inconsistent vector field.

This is a very powerful GT metric because it measures **non-conservativeness** of f(x).


