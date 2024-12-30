export create_matrix
export create_vector

export derivative
export gradient
export jacobian
export hessian

const global REGISTERED_SYMBOLS = Dict{String,Tensor}()

function create_matrix(name::String)
    T = Tensor(name, Upper(get_next_letter()), Lower(get_next_letter()))

    REGISTERED_SYMBOLS[name] = T

    return T
end

function create_vector(name::String)
    T = Tensor(name, Upper(get_next_letter()))

    REGISTERED_SYMBOLS[name] = T

    return T
end

function derivative(expr, wrt::String)
    ∂ = Tensor(wrt)

    if wrt ∈ keys(REGISTERED_SYMBOLS)
        for index ∈ REGISTERED_SYMBOLS[wrt].indices
            push!(∂.indices, same_to(index, get_next_letter()))
        end
    else
        throw(DomainError(wrt, "Unknown symbol $wrt"))
    end

    D = diff(expr, ∂)

    return evaluate(D)
end

function gradient(expr, wrt::String)
    free_indices = get_free_indices(evaluate(expr))

    if !isempty(free_indices)
        throw(DomainError(evaluate(expr), "Input is not a scalar"))
    end

    if wrt ∈ keys(REGISTERED_SYMBOLS)
        if length(REGISTERED_SYMBOLS[wrt].indices) != 1
            throw(DomainError(wrt, "\"$wrt\" is not a vector"))
        end
    else
        throw(DomainError(wrt, "Unknown symbol \"$wrt\""))
    end

    D = derivative(expr, wrt)
    gradient = evaluate(D')

    return gradient
end

function jacobian(expr, wrt::String)
    free_indices = get_free_indices(evaluate(expr))

    if length(free_indices) != 1 || typeof(free_indices[1]) != Upper
        throw(DomainError(evaluate(expr), "Input is not a column vector"))
    end

    if wrt ∈ keys(REGISTERED_SYMBOLS)
        if length(REGISTERED_SYMBOLS[wrt].indices) != 1
            throw(DomainError(wrt, "\"$wrt\" is not a vector"))
        end
    else
        throw(DomainError(wrt, "Unknown symbol \"$wrt\""))
    end

    D = derivative(expr, wrt)

    return D
end

function hessian(expr, wrt::String)
    free_indices = get_free_indices(evaluate(expr))

    if !isempty(free_indices)
        throw(DomainError(evaluate(expr), "Input is not a scalar"))
    end

    if wrt ∈ keys(REGISTERED_SYMBOLS)
        if length(REGISTERED_SYMBOLS[wrt].indices) != 1
            throw(DomainError(wrt, "\"$wrt\" is not a vector"))
        end
    else
        throw(DomainError(wrt, "Unknown symbol \"$wrt\""))
    end

    D = derivative(expr, wrt)
    g = evaluate(D')
    H = derivative(g, wrt)

    return H
end

function to_std_string(arg::Tensor, transpose::Bool)
    if transpose
        return arg.id * "ᵀ"
    else
        return arg.id
    end
end

# function to_std_string(arg::KrD, transpose::Bool)
#     if length(arg.indices) == 2
#         if typeof(arg.indices[1]) == Upper && typeof(arg.indices[2]) == Lower
#             return "I"
#         end

#         if typeof(arg.indices[1]) == Lower && typeof(arg.indices[2]) == Upper
#             return "I"
#         end
#     end

#     throw_not_std()
# end

function to_std_string(arg::Real, transpose::Bool)
    return to_string(arg)
end

function to_std_string(arg::Negate, transpose::Bool)
    return "-" * to_std_string(arg.arg, transpose)
end

function to_std_string(arg::Sin, transpose::Bool)
    return "sin(" * to_std_string(arg.arg, transpose) * ")"
end

function to_std_string(arg::Cos, transpose::Bool)
    return "cos(" * to_std_string(arg.arg, transpose) * ")"
end

function parenthesize_std(arg)
    return to_std_string(arg)
end

function parenthesize_std(arg::BinaryOperation{Add})
    return "(" * to_std_string(arg) * ")"
end

function get_sym(arg::Tensor)
    return arg.id
end

function throw_not_std()
    throw(DomainError("Cannot write expression in standard notation"))
end

function to_std_string(::Add)
    return "+"
end

function to_std_string(::Sub)
    return "-"
end

function vectorin(a, b)
    s = Set(b)

    return [e ∈ s for e ∈ a]
end

function to_std_string(arg::BinaryOperation{Op}, transpose::Bool) where {Op<:AdditiveOperation}
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert all(vectorin(arg1_ids, arg2_ids)) && all(vectorin(arg2_ids, arg1_ids))

    if length(arg1_ids) > 2
        throw_not_std()
    end

    transpose_left = collect(reverse(arg1_ids)) == arg2_ids
    transpose_left = transpose ? !transpose_left : transpose_left

    return to_std_string(arg.arg1, transpose_left) * " " * to_std_string(Op()) * " " * to_std_string(arg.arg2, transpose)
end

function collect_terms(arg::BinaryOperation{Mult})
    return [collect_terms(arg.arg1); collect_terms(arg.arg2)]
end

function collect_terms(arg)
    return [arg]
end

function is_index_order_same(terms, index_order)
    indices = reduce(vcat, get_free_indices.(terms))

    # if length(terms) == 1
    #     is_valid = true
    #     for idx ∈ indices
    #         if idx ∈ index_order

    i = 1

    for index ∈ indices
        if i <= length(index_order) && index == index_order[i]
            i += 1
        elseif index ∈ index_order
            return false
        end
    end

    if i == 1
        return false
    end

    return true
end

function is_trace2(arg)
    terms = collect_terms(arg)
    # TODO: For completeness one should actually check all contractions and make sure that there
    # are no vector'-vector products
    return all(length.(get_free_indices.(terms)) .== 2) && isempty(get_free_indices(arg))
end

function transpose_sequence(seq)
    full_term = reduce(*, seq)

    full_term_t = evaluate(full_term')

    return collect_terms(full_term_t)
end

function to_std_string(arg, transpose::Bool = false)
    target_indices = get_free_indices(arg)

    terms = collect_terms(arg)
    remaining = Any[t for t ∈ terms]

    trace = is_trace2(arg)

    ordered_args = []

    run = true

    for i ∈ eachindex(remaining)
        if isnothing(remaining[i])
            continue
        end
        if isempty(ordered_args)
            push!(ordered_args, remaining[i])
            remaining[i] = nothing
            continue
        end
        if all(isnothing.(remaining))
            run = false
        end

        term = remaining[i]
        term_indices = get_free_indices(term)

        for fixed ∈ (ordered_args[1], ordered_args[end])
            fixed_indices = get_free_indices(fixed)

            if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[1]
                pushfirst!(ordered_args, term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[end]
                push!(ordered_args, term)
                remaining[i] = nothing
                break
            end

            if  flip(term_indices[end]) == fixed_indices[end]
                if typeof(fixed_indices[end]) == Lower
                    push!(ordered_args, term)
                else
                    pushfirst!(ordered_args, term)
                end
                remaining[i] = nothing
                break
            end

            if flip(term_indices[1]) == fixed_indices[1]
                if typeof(fixed_indices[1]) == Lower
                    push!(ordered_args, term)
                else
                    pushfirst!(ordered_args, term)
                end
                remaining[i] = nothing
                break
            end
        end
                
        
    end

    argstr = ""

    if length(target_indices) == 2
        if typeof(target_indices[1]) == Upper && typeof(target_indices[2]) == Lower || typeof(target_indices[1]) == Lower && typeof(target_indices[2]) == Upper
            has_row_vector = false
            for term ∈ ordered_args
                ids = get_free_indices(term)
                if length(ids) == 1 && typeof(ids[1]) == Lower
                    has_row_vector = true
                end
            end

            if has_row_vector
                ordered_args = collect(reverse(ordered_args))
            end
        end
    elseif length(target_indices) == 1
        if typeof(target_indices[1]) == Upper
            has_row_vector = false
            for term ∈ ordered_args
                ids = get_free_indices(term)
                if length(ids) == 1 && typeof(ids[1]) == Lower
                    has_row_vector = true
                end
            end

            if has_row_vector
                ordered_args = collect(reverse(ordered_args))
            end
        elseif typeof(target_indices[1]) == Lower
            has_col_vector = false
            for term ∈ ordered_args
                ids = get_free_indices(term)
                if length(ids) == 1 && typeof(ids[1]) == Upper
                    has_col_vector = true
                end
            end

            if has_col_vector
                ordered_args = collect(reverse(ordered_args))
            end
        end
    else
        throw_not_std()
    end

    for t ∈ ordered_args
        indices = get_free_indices(t)

        if typeof(indices[end]) == Lower
            if length(indices) == 2
                argstr *= to_std_string(t, transpose)
            elseif length(indices) == 1
                argstr *= to_std_string(t, !transpose)
            else
                throw_not_std()
            end
        else
            argstr *= to_std_string(t, transpose)
        end
    end

    if trace
        argstr = "tr(" * argstr * ")"
    end

    return argstr
end
