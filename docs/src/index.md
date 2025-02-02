# DiffMatic.jl

Documentation for DiffMatic.jl

```@docs
@matrix(ids...)
@vector(ids...)
@scalar(ids...)
derivative(expr, wrt::DiffMatic.Tensor)
gradient(expr, wrt::DiffMatic.Tensor)
jacobian(expr, wrt::DiffMatic.Tensor)
hessian(expr, wrt::DiffMatic.Tensor)
to_std_string(expr)
```
