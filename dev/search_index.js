var documenterSearchIndex = {"docs":
[{"location":"#Judi.jl","page":"Judi.jl","title":"Judi.jl","text":"","category":"section"},{"location":"","page":"Judi.jl","title":"Judi.jl","text":"Documentation for Judi.jl","category":"page"},{"location":"","page":"Judi.jl","title":"Judi.jl","text":"@matrix(ids...)\n@vector(ids...)\n@scalar(ids...)\nderivative(expr, wrt::Tensor)\ngradient(expr, wrt::Tensor)\njacobian(expr, wrt::Tensor)\nhessian(expr, wrt::Tensor)","category":"page"},{"location":"#Judi.@matrix-Tuple","page":"Judi.jl","title":"Judi.@matrix","text":"@matrix(ids...)\n\nCreate one or more matrices. Example:\n\n@matrix A X\n\n\n\n\n\n","category":"macro"},{"location":"#Judi.@vector-Tuple","page":"Judi.jl","title":"Judi.@vector","text":"@vector(ids...)\n\nCreate one or more vectors. Example:\n\n@vector x y\n\n\n\n\n\n","category":"macro"},{"location":"#Judi.@scalar-Tuple","page":"Judi.jl","title":"Judi.@scalar","text":"@scalar(ids...)\n\nCreate one or more scalars. Example:\n\n@scalar a β\n\n\n\n\n\n","category":"macro"},{"location":"#Judi.derivative-Tuple{Any, Tensor}","page":"Judi.jl","title":"Judi.derivative","text":"derivative(expr, wrt::Tensor)\n\nCompute the derivative of expr with respect to wrt.\n\n\n\n\n\n","category":"method"},{"location":"#Judi.gradient-Tuple{Any, Tensor}","page":"Judi.jl","title":"Judi.gradient","text":"gradient(expr, wrt::Tensor)\n\nCompute the gradient of expr with respect to wrt. expr must be a scalar and wrt a vector.\n\n\n\n\n\n","category":"method"},{"location":"#Judi.jacobian-Tuple{Any, Tensor}","page":"Judi.jl","title":"Judi.jacobian","text":"jacobian(expr, wrt::Tensor)\n\nCompute the jacobian of expr with respect to wrt. expr must be a column vector and wrt a vector.\n\n\n\n\n\n","category":"method"},{"location":"#Judi.hessian-Tuple{Any, Tensor}","page":"Judi.jl","title":"Judi.hessian","text":"hessian(expr, wrt::Tensor)\n\nCompute the hessian of expr with respect to wrt. expr must be a scalar and wrt a vector.\n\n\n\n\n\n","category":"method"}]
}
