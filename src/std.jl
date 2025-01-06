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

# function _to_std_string(arg::Tensor)
#     return arg.id
# end

function _to_std_string(arg::Tensor)
    ids = get_free_indices(arg)

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

# function to_std_string(arg::KrD)
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

function _to_std_string(arg::Real)
    return to_string(arg)
end

function _to_std_string(arg::Negate)
    return "-" * _to_std_string(arg.arg, transpose)
end

function _to_std_string(arg::Sin)
    return "sin(" * _to_std_string(arg.arg, transpose) * ")"
end

function _to_std_string(arg::Cos)
    return "cos(" * _to_std_string(arg.arg, transpose) * ")"
end

function parenthesize_std(arg)
    return _to_std_string(arg)
end

function parenthesize_std(arg::BinaryOperation{Op}) where {Op <: AdditiveOperation}
    return "(" * _to_std_string(arg) * ")"
end

function get_sym(arg::Tensor)
    return arg.id
end

function throw_not_std()
    throw(DomainError("Cannot write expression in standard notation"))
end

function _to_std_string(::Add)
    return "+"
end

function _to_std_string(::Sub)
    return "-"
end

function vectorin(a, b)
    s = Set(b)

    return [e ∈ s for e ∈ a]
end

function _to_std_string(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return _to_std_string(arg.arg1) * " " * _to_std_string(Op()) * " " * _to_std_string(arg.arg2)
end

function _to_std_string(arg::BinaryOperation{Mult})
    return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
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

function _transpose(term)
    indices = get_free_indices(term)

    for index ∈ indices
        tmp_index = flip_to(index, get_next_letter())
        term = evaluate(update_index(reshape(term, index, tmp_index), tmp_index, flip(index)))
    end

    return term
end

function transpose_sequence(seq)
    return collect(reverse(map(_transpose, seq)))
end

function to_matrix(term)
    ids = get_free_indices(term)
    return Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter))
end

function to_matrix_t(term)
    return _transpose(to_matrix(term))
end

function to_column_vector(term)
    ids = get_free_indices(term)
    return Tensor(term.id, Upper(ids[1]))
end

function to_row_vector(term)
    return _transpose(to_column_vector(term))
end

# function to_matrix(term, left_index = nothing, right_index = nothing)
#     ids = get_free_indices(term)

#     if isnothing(left_index) && isnothing(right_index)
#         return Tensor(term.id, Upper(ids[1]), Lower(ids[2]))
#     end

#     if isnothing(left_index)
#         if right_index == ids[2]
#             return Tensor(term.id, flip_to(ids[2], ids[1]), ids[2])
#         end

#         if right_index == ids[1]
#             return Tensor(term.id, )

# end

function to_standard(term::Tensor, upper_index = nothing, lower_index = nothing)
    ids = get_free_indices(term)

    if length(ids) == 2
        if isnothing(upper_index) && isnothing(lower_index)
            return Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter))
        end

        if isnothing(upper_index)
            if ids[2].letter == lower_index
                return Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter))
            end

            if ids[1].letter == lower_index
                return Tensor(term.id, Lower(ids[1].letter), Upper(ids[2].letter))
            end
        end

        if isnothing(lower_index)
            if ids[2].letter == upper_index
                return Tensor(term.id, Lower(ids[1].letter), Upper(ids[2].letter))
            end

            if ids[1].letter == upper_index
                return Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter))
            end
        end

        if upper_index == ids[1].letter && lower_index == ids[2].letter
            return Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter))
        end

        if upper_index == ids[2].letter && lower_index == ids[1].letter
            return Tensor(term.id, Lower(ids[1].letter), Upper(ids[2].letter))
        end
    elseif length(ids) == 1
        @assert !(!isnothing(upper_index) && !isnothing(lower_index))
        if isnothing(upper_index) && isnothing(lower_index)
            return Tensor(term.id, Upper(ids[1].letter))
        end

        if isnothing(upper_index) && ids[1].letter == lower_index
            return Tensor(term.id, Lower(ids[1].letter))
        end

        if isnothing(lower_index) && ids[1].letter == upper_index
            return Tensor(term.id, Upper(ids[1].letter))
        end
    end

    @assert false
end

function to_standard(arg::BinaryOperation{Op}, upper_index = nothing, lower_index = nothing) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(to_standard(arg.arg1, upper_index, lower_index), to_standard(arg.arg2, upper_index, lower_index))
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

function to_standard(arg, upper_index = nothing, lower_index = nothing)
    target_indices = get_free_indices(arg)

    if length(target_indices) > 2
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
            end

            break
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
        else
            throw_not_std()
        end
    end

    @assert !isempty(ordered_args)

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

            if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[1]
                std_term = to_standard(term, nothing, term_indices[end].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                pushfirst!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[end]) == Upper && flip(term_indices[end]) == fixed_indices[1]
                std_term = to_standard(term, term_indices[end].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[end]
                std_term = to_standard(term, term_indices[1].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[1]) == Lower && flip(term_indices[1]) == fixed_indices[end]
                std_term = to_standard(term, nothing, term_indices[1].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                pushfirst!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[end]) == Upper && flip(term_indices[end]) == fixed_indices[end]
                std_term = to_standard(term, term_indices[end].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[end]
                std_term = to_standard(term, nothing, term_indices[end].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                pushfirst!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[1]
                std_term = to_standard(term, term_indices[1].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                push!(ordered_args, std_term)
                remaining[i] = nothing
                break
            end

            if typeof(term_indices[1]) == Lower && flip(term_indices[1]) == fixed_indices[1]
                std_term = to_standard(term, nothing, term_indices[1].letter)
                flipped_indices[std_term] = get_flipped(std_term, term)
                pushfirst!(ordered_args, std_term)
                remaining[i] = nothing
                break
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

    @assert length(ordered_expr_ids) == length(target_indices)

    return standardized_term
end

function to_std_string(arg)
    free_indices = get_free_indices(arg)

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
        throw_not_std()
    end

    # TODO: rename
    trace = is_trace2(arg)

    argstr = _to_std_string(standardized)

    if trace
        argstr = "tr(" * argstr * ")"
    end

    return argstr
end
