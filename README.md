# MatrixDiff.jl

[![Run tests](https://github.com/asterycs/MatrixDiff.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/asterycs/MatrixDiff.jl/actions/workflows/CI.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Symbolic differentiation of vector/matrix/tensor expressions in Julia

This library implements symbolic matrix differentiation. The implementation is based on the ideas presented in

> S. Laue, M. Mitterreiter, and J. Giesen.  
> Computing Higher Order Derivatives of Matrix and Tensor Expressions, NeurIPS 2018.

### Current status

This library is currently in prototype stage. Many details are still missing and the behavior of the existing parts are changing frequently. Most notably, a user friendly API is still missing. The user currently has to know a bit of Ricci notation in order to use it.
