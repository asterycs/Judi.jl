# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

export @matrix
export @vector
export @scalar

export derivative
export gradient
export jacobian
export hessian

function create_matrix(name::String)
    T = Tensor(name, Upper(get_next_letter()), Lower(get_next_letter()))

    return T
end

function create_vector(name::String)
    T = Tensor(name, Upper(get_next_letter()))

    return T
end

function create_scalar(name::String)
    T = Tensor(name)

    return T
end

"""
    @matrix(ids...)

Create one or more matrices. Example:
```jldoctest; output=false
@matrix A X

# output

X¹⁰²₁₀₃
```
"""
macro matrix(ids...)
    # From https://discourse.julialang.org/t/macro-question-create-variables/90160
    syms = (:($(esc(id)) = create_matrix($(string(id)))) for id ∈ ids)
    return Expr(:block, syms...)
end

"""
    @vector(ids...)

Create one or more vectors. Example:
```jldoctest; output=false
@vector x y

# output

y¹⁰⁵
```
"""
macro vector(ids...)
    syms = (:($(esc(id)) = create_vector($(string(id)))) for id ∈ ids)
    return Expr(:block, syms...)
end

"""
    @scalar(ids...)

Create one or more scalars. Example:
```jldoctest; output=false
@scalar a β

# output

β
```
"""
macro scalar(ids...)
    syms = (:($(esc(id)) = create_scalar($(string(id)))) for id ∈ ids)
    return Expr(:block, syms...)
end

"""
    derivative(expr, wrt::Tensor)

Compute the derivative of `expr` with respect to `wrt`.
"""
function derivative(expr, wrt::Tensor)
    ∂ = Tensor(wrt.id)

    for index ∈ wrt.indices
        push!(∂.indices, same_to(index, get_next_letter()))
    end

    D = diff(expr, ∂)

    return evaluate(D)
end

"""
    gradient(expr, wrt::Tensor)

Compute the gradient of `expr` with respect to `wrt`. `expr` must be a scalar and `wrt` a vector.
"""
function gradient(expr, wrt::Tensor)
    free_indices = get_free_indices(evaluate(expr))

    if !isempty(free_indices)
        throw(DomainError(evaluate(expr), "Input is not a scalar"))
    end

    if length(wrt.indices) != 1
        throw(DomainError(wrt, "\"$wrt\" is not a vector"))
    end

    D = derivative(expr, wrt)
    gradient = evaluate(D')

    return gradient
end

"""
    jacobian(expr, wrt::Tensor)

Compute the jacobian of `expr` with respect to `wrt`. `expr` must be a column vector and `wrt` a vector.
"""
function jacobian(expr, wrt::Tensor)
    free_indices = get_free_indices(evaluate(expr))

    if length(free_indices) != 1 || typeof(free_indices[1]) != Upper
        throw(DomainError(evaluate(expr), "Input is not a column vector"))
    end

    if length(wrt.indices) != 1
        throw(DomainError(wrt, "\"$wrt\" is not a vector"))
    end

    D = derivative(expr, wrt)

    return D
end

"""
    hessian(expr, wrt::Tensor)

Compute the hessian of `expr` with respect to `wrt`. `expr` must be a scalar and `wrt` a vector.
"""
function hessian(expr, wrt::Tensor)
    free_indices = get_free_indices(evaluate(expr))

    if !isempty(free_indices)
        throw(DomainError(evaluate(expr), "Input is not a scalar"))
    end

    if length(wrt.indices) != 1
        throw(DomainError(wrt, "\"$wrt\" is not a vector"))
    end

    D = derivative(expr, wrt)
    g = evaluate(D')
    H = derivative(g, wrt)

    return H
end

function _to_std_string(arg::Tensor)
    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return arg.id
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return arg.id * "ᵀ"
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return arg.id
        elseif typeof(ids[1]) == Lower
            return arg.id * "ᵀ"
        end
    end

    throw_not_std()
end

function _to_std_string(arg::KrD)
    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return "I"
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return "Iᵀ"
        end
    end

    throw_not_std()
end

function _to_std_string(arg::Zero)
    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return "mat(0)"
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return "mat(0)ᵀ"
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return "vec(0)"
        elseif typeof(ids[1]) == Lower
            return "vec(0)ᵀ"
        end
    end

    throw_not_std()
end

function _to_std_string(arg::Real)
    return to_string(arg)
end

function _to_std_string(arg::Negate)
    return "-" * _to_std_string(arg.arg)
end

function _to_std_string(arg::Sin)
    return "sin(" * _to_std_string(arg.arg) * ")"
end

function _to_std_string(arg::Cos)
    return "cos(" * _to_std_string(arg.arg) * ")"
end

function _to_std_string(::Add)
    return "+"
end

function _to_std_string(::Sub)
    return "-"
end

function _to_std_string(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return _to_std_string(arg.arg1) * " " * _to_std_string(Op()) * " " * _to_std_string(arg.arg2)
end

function _to_std_string(arg::BinaryOperation{Mult})
    # TODO: Create separate type for elementwise
    if is_elementwise_multiplication(arg.arg1, arg.arg2)
        arg1_indices,arg2_indices = get_free_indices.((arg.arg1, arg.arg2))

        if length(arg1_indices) == length(arg2_indices)
            return parenthesize_std(arg.arg1) * " ⊙ " * parenthesize_std(arg.arg2)
        elseif length(arg1_indices) == 1 && length(arg2_indices) == 2
            if typeof(arg1_indices[1]) == Upper
                return "diag(" * _to_std_string(arg.arg1) * ")" * parenthesize_std(arg.arg2)
            else # typeof(arg1_indices[1]) == Lower
                return parenthesize_std(arg.arg2) * " diag(" * _to_std_string(arg.arg1) * ")"
            end
        elseif length(arg1_indices) == 2 && length(arg2_indices) == 1
            if typeof(arg2_indices[1]) == Upper
                return "diag(" * _to_std_string(arg.arg2) * ")" * parenthesize_std(arg.arg1)
            else # typeof(arg2_indices[1]) == Lower
                return parenthesize_std(arg.arg1) * " diag(" * _to_std_string(arg.arg2) * ")"
            end
        end
    end

    return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
end


function parenthesize_std(arg)
    return _to_std_string(arg)
end

function parenthesize_std(arg::BinaryOperation{Op}) where {Op <: AdditiveOperation}
    return "(" * _to_std_string(arg) * ")"
end

function parenthesize_std(arg::BinaryOperation{Mult})
    # TODO: Create separate type for elementwise
    if is_elementwise_multiplication(arg.arg1, arg.arg2)
        return "(" * _to_std_string(arg.arg1) * " ⊙ " * _to_std_string(arg.arg2) * ")"
    end

    return _to_std_string(arg)
end

function throw_not_std()
    throw(DomainError("Cannot write expression in standard notation"))
end

function collect_terms(arg::BinaryOperation{Mult})
    # TODO: Create separate type for elementwise
    if is_elementwise_multiplication(arg.arg1, arg.arg2)
        return [arg]
    end

    return [collect_terms(arg.arg1); collect_terms(arg.arg2)]
end

function collect_terms(arg)
    return [arg]
end

function is_trace(arg)
    terms = collect_terms(arg)

    if length(terms) == 1
        ids = get_indices(arg)
        free_ids = get_free_indices(arg)

        if length(ids) == 2 && isempty(free_ids)
            return true
        else
            return false
        end
    end

    return all(length.(get_free_indices.(terms)) .== 2) && isempty(get_free_indices(arg))
end

function reshape(term::Tensor, indices::LowerOrUpperIndex...)
    return Tensor(term.id, indices...)
end

function reshape(term::Zero, indices::LowerOrUpperIndex...)
    return Zero(indices...)
end

function reshape(term::KrD, indices::LowerOrUpperIndex...)
    return KrD(indices...)
end

function to_standard(term::Op, upper_index = nothing, lower_index = nothing) where {Op<:UnaryOperation}
    return Op(to_standard(term.arg, upper_index, lower_index))
end

function to_standard(term, upper_index = nothing, lower_index = nothing)
    ids = get_free_indices(term)

    if length(ids) == 2
        if isnothing(upper_index) && isnothing(lower_index)
            return reshape(term, Upper(ids[1].letter), Lower(ids[2].letter))
        end

        if isnothing(upper_index)
            if ids[2].letter == lower_index
                return reshape(term, Upper(ids[1].letter), Lower(ids[2].letter))
            end

            if ids[1].letter == lower_index
                return reshape(term, Lower(ids[1].letter), Upper(ids[2].letter))
            end
        end

        if isnothing(lower_index)
            if ids[2].letter == upper_index
                return reshape(term, Lower(ids[1].letter), Upper(ids[2].letter))
            end

            if ids[1].letter == upper_index
                return reshape(term, Upper(ids[1].letter), Lower(ids[2].letter))
            end
        end

        if upper_index == ids[1].letter && lower_index == ids[2].letter
            return reshape(term, Upper(ids[1].letter), Lower(ids[2].letter))
        end

        if upper_index == ids[2].letter && lower_index == ids[1].letter
            return reshape(term, Lower(ids[1].letter), Upper(ids[2].letter))
        end
    elseif length(ids) == 1
        @assert !(!isnothing(upper_index) && !isnothing(lower_index))
        if isnothing(upper_index) && isnothing(lower_index)
            return reshape(term, Upper(ids[1].letter))
        end

        if isnothing(upper_index) && ids[1].letter == lower_index
            return reshape(term, Lower(ids[1].letter))
        end

        if isnothing(lower_index) && ids[1].letter == upper_index
            return reshape(term, Upper(ids[1].letter))
        end
    end

    # No free indices - check if this is this a trace
    if is_trace(term)
        ids = get_indices(term)

        if typeof(ids[1]) == Upper
            return reshape(term, ids[1], Lower(ids[2].letter))
        else typeof(ids[2]) == Lower
            return reshape(term, ids[1], Upper(ids[2].letter))
        end
    end

    throw_not_std()
end

function to_standard(arg::BinaryOperation{Op}, upper_index = nothing, lower_index = nothing) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(to_standard(arg.arg1, upper_index, lower_index), to_standard(arg.arg2, upper_index, lower_index))
end

function to_standard(arg::Real, upper_index = nothing, lower_index = nothing)
    return arg
end

function get_flipped(new_term, old_term)
    new_ids = get_free_indices(new_term)
    old_ids = get_free_indices(old_term)

    flipped = Dict()

    for (l,r) ∈ zip(new_ids, old_ids)
        @assert l.letter == r.letter

        if typeof(l) != typeof(r)
            flipped[r] = l
        end
    end

    return flipped
end

function was_flipped(index, flips)
    if flip(index) ∈ keys(flips)
        return true
    end

    return false
end

function to_standard(arg::BinaryOperation{Mult}, upper_index = nothing, lower_index = nothing)
    target_indices = unique(get_free_indices(arg))

    if length(target_indices) > 2
        throw_not_std()
    end

    # TODO: Create separate type for elementwise
    if is_elementwise_multiplication(arg.arg1, arg.arg2)
        upper = nothing
        lower = nothing
        if !isempty(target_indices)
            if target_indices[1].letter == upper_index
                upper = target_indices[1].letter
            end
            if target_indices[1].letter == lower_index
                lower = target_indices[1].letter
            end
        end
        if length(target_indices) > 1
            if target_indices[2].letter == upper_index
                upper = target_indices[2].letter
            end
            if target_indices[2].letter == lower_index
                lower = target_indices[2].letter
            end
        end

        arg1_indices,arg2_indices = get_free_indices.((arg.arg1, arg.arg2))

        if length(arg1_indices) == length(arg2_indices)
            return BinaryOperation{Mult}(to_standard(arg.arg1, upper, lower), to_standard(arg.arg2, upper, lower))
        elseif length(arg1_indices) == 1 && length(arg2_indices) == 2
            if typeof(arg1_indices[1]) == Upper
                return BinaryOperation{Mult}(to_standard(arg.arg1, upper), to_standard(arg.arg2, upper, lower))
            else # typeof(arg1_indices[1]) == Lower
                return BinaryOperation{Mult}(to_standard(arg.arg1, nothing, lower), to_standard(arg.arg2, upper, lower))
            end
        elseif length(arg1_indices) == 2 && length(arg2_indices) == 1
            if typeof(arg2_indices[1]) == Upper
                return BinaryOperation{Mult}(to_standard(arg.arg1, upper, lower), to_standard(arg.arg2, upper))
            elseif typeof(arg2_indices[1]) == Lower
                return BinaryOperation{Mult}(to_standard(arg.arg1, upper, lower), to_standard(arg.arg2, nothing, lower))
            end
        end

        throw_not_std()
    end

    terms = collect_terms(arg)
    remaining = Any[t for t ∈ terms]

    for term ∈ terms
        if length(get_free_indices(term)) > 2
            throw_not_std()
        end
    end

    flipped_indices = Dict()
    ordered_args = []

    for i ∈ eachindex(remaining)
        if isnothing(remaining[i])
            continue
        end

        term = remaining[i]
        ids = get_free_indices(term)
        if length(ids) == 2
            if ids[1].letter == upper_index || ids[2].letter == lower_index
                std_term = nothing
                if !isnothing(upper_index) && !isnothing(lower_index)
                    std_term = to_standard(term, ids[1].letter, ids[2].letter)
                elseif isnothing(upper_index)
                    std_term = to_standard(term, nothing, ids[2].letter)
                else
                    std_term = to_standard(term, ids[1].letter)
                end
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            elseif ids[1].letter == lower_index || ids[2].letter == upper_index
                std_term = nothing
                if !isnothing(upper_index) && !isnothing(lower_index)
                    std_term = to_standard(term, ids[2].letter, ids[1].letter)
                elseif isnothing(upper_index)
                    std_term = to_standard(term, nothing, ids[1].letter)
                else
                    std_term = to_standard(term, ids[2].letter)
                end
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end
        elseif length(ids) == 1
            std_term = nothing
            if ids[1].letter == lower_index
                std_term = to_standard(term, nothing, ids[1].letter)
            elseif ids[1].letter == upper_index
                std_term = to_standard(term, ids[1].letter)
            end
            if !isnothing(std_term)
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end
        elseif isempty(ids)
            # save scalars for later
        else
            throw_not_std()
        end
    end

    if isnothing(upper_index) && isnothing(lower_index)
        @assert isempty(ordered_args)

        std_term = to_standard(remaining[1])
        push!(ordered_args, std_term)
        flipped_indices[std_term] = get_flipped(std_term, remaining[1])
        remaining[1] = nothing
    end

    @assert !isempty(ordered_args)

    repeat_try_add = true

    # TODO: Refactor this loop
    while repeat_try_add
        term_was_added = false

        for i ∈ eachindex(remaining)
            if isnothing(remaining[i])
                continue
            end

            if all(isnothing.(remaining))
                break
            end

            term = remaining[i]

            for fixed ∈ (ordered_args[1], ordered_args[end])
                fixed_indices = get_free_indices(fixed)
                term_indices = get_free_indices(term)

                for i ∈ eachindex(term_indices)
                    if was_flipped(term_indices[i], flipped_indices[fixed])
                        term_indices[i] = flip(term_indices[i])
                    end
                end

                if isempty(term_indices)
                    std_term = to_standard(term)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    pushfirst!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                ### Contractions
                ################
                if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[1]
                    std_term = to_standard(term, nothing, term_indices[end].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    pushfirst!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[end]) == Upper && flip(term_indices[end]) == fixed_indices[1]
                    std_term = to_standard(term, term_indices[end].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    push!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[end]
                    std_term = to_standard(term, term_indices[1].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    push!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[1]) == Lower && flip(term_indices[1]) == fixed_indices[end]
                    std_term = to_standard(term, nothing, term_indices[1].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    pushfirst!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[end]) == Upper && flip(term_indices[end]) == fixed_indices[end]
                    std_term = to_standard(term, term_indices[end].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    push!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[end]
                    std_term = to_standard(term, nothing, term_indices[end].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    pushfirst!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[1]
                    std_term = to_standard(term, term_indices[1].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    push!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end

                if typeof(term_indices[1]) == Lower && flip(term_indices[1]) == fixed_indices[1]
                    std_term = to_standard(term, nothing, term_indices[1].letter)
                    flipped_indices[std_term] = get_flipped(std_term, term)
                    pushfirst!(ordered_args, std_term)
                    remaining[i] = nothing
                    term_was_added = true
                    break
                end
            end
        end

        if !term_was_added
            if all(isnothing.(remaining))
                repeat_try_add = false
            else
                throw_not_std()
            end
        end
    end

    @assert all(isnothing.(remaining))

    standardized_term = nothing

    for t ∈ ordered_args
        if isnothing(standardized_term)
            standardized_term = t
            continue
        end

        standardized_term = BinaryOperation{Mult}(standardized_term, t)
    end

    ordered_expr_ids = get_free_indices(standardized_term)

    @assert length(unique(ordered_expr_ids)) == length(target_indices)

    return standardized_term
end

function to_std_string(arg)
    free_indices = unique(get_free_indices(arg))

    standardized = if length(free_indices) == 2
        if typeof(free_indices[1]) == Upper && typeof(free_indices[2]) == Lower
            to_standard(arg, free_indices[1].letter, free_indices[2].letter)
        elseif typeof(free_indices[1]) == Lower && typeof(free_indices[2]) == Upper
            to_standard(arg, free_indices[2].letter, free_indices[1].letter)
        else
            throw_not_std()
        end
    elseif length(free_indices) == 1
        if typeof(free_indices[1]) == Lower
            to_standard(arg, nothing, free_indices[1].letter)
        elseif typeof(free_indices[1]) == Upper
            to_standard(arg, free_indices[1].letter)
        else
            throw_not_std()
        end
    else
        to_standard(arg)
    end

    trace = is_trace(arg)

    argstr = _to_std_string(standardized)

    if trace
        argstr = "tr(" * argstr * ")"
    end

    return argstr
end
