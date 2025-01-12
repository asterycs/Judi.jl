# Judi.jl

_<ins>J</ins>ust a <ins>di</ins>fferentiation library_

[![Run tests](https://github.com/asterycs/Judi.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/asterycs/Judi.jl/actions/workflows/CI.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Symbolic differentiation of vector/matrix/tensor expressions in Julia

#### This package is still under development - there be dragons. Issues are welcome.

### Example

Create a matrix and a vector:

```julia
using Judi

@matrix A
@vector x
```
Create an expression:
```julia
expr = x' * A * x
```
The variable `expr` now contains an internal representation of the expression `x' * A * x`.

Compute the gradient and the Hessian with respect to the vector `x`.
```julia
g = gradient(expr, x)
H = hessian(expr, x)
```
Convert the gradient and the Hessian into standard notation using `to_std_string`:
```julia
to_std_string(g) # "Aᵀx + Ax"
to_std_string(H) # "Aᵀ + A"
```

Jacobians can be computed with `jacobian`:

```julia
to_std_string(jacobian(A * x, x)) # "A"
```

The method `derivative` can be used to compute arbitrary derivatives.

```julia
to_std_string(derivative(tr(A), A)) # "I"
```
The method `to_std_string` will throw an exception when given an expression that that cannot be converted to
standard notation.

### Supported operators

`tr`, `sin`, `cos`, `+`, `-`, `*`, `'`, `.*`

### Installation

This library is not yet published in the general registry. To install it directly from Github:

```julia
using Pkg; Pkg.add("https://github.com/asterycs/Judi.jl.git")
```

### Acknowledgements

The implementation is based on the ideas presented in

> S. Laue, M. Mitterreiter, and J. Giesen.
> Computing Higher Order Derivatives of Matrix and Tensor Expressions, NeurIPS 2018.
