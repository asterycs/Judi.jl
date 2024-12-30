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

    # TODO: Should instead make sure that the covariant dimension is last for matrices
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

# function transpose_sequence(seq)
#     full_term = reduce(*, seq)

#     full_term_t = evaluate(full_term')

#     return collect_terms(full_term_t)
# end

struct StdizedTerm
    term
    transpose::Bool
end

function get_free_indices(term::StdizedTerm)
    ids = get_free_indices(term.term)

    if term.transpose
        return collect(reverse(map(flip, ids)))
    end

    return ids
end

function transpose_term(term::StdizedTerm)
    return StdizedTerm(term.term, !term.transpose)
end

function transpose_sequence(seq)
    return collect(reverse(map(transpose_term, seq)))
end

function to_std_string(arg, transpose::Bool = false)
    target_indices = get_free_indices(arg)

    terms = collect_terms(arg)
    remaining = Any[t for t ∈ terms]

    trace = is_trace2(arg)
    # Trace prperty: order of the terms does not matter.

    ordered_args = StdizedTerm[]

    # for i ∈ eachindex(remaining)
    #     arg = remaining[i]
    #     arg_ids = get_free_indices(arg)
    #     same_ids = intersect(arg_ids, target_indices)

    #     @show arg
    #     @show ordered_args
    #     @show target_indices
    #     @show same_ids

    #     if !isempty(same_ids)
    #         if is_index_order_same([arg; ordered_args], target_indices)
    #             pushfirst!(ordered_args, arg)
    #             remaining[i] = nothing
    #         elseif is_index_order_same([ordered_args; arg], target_indices)
    #             push!(ordered_args, arg)
    #             remaining[i] = nothing
    #         else
    #             println("what happened")
    #             return
    #         end
    #     end
    # end

    run = true

    # # while run
    #     for i ∈ eachindex(remaining)
    #         if isnothing(remaining[i])
    #             continue
    #         end
    #         if isempty(ordered_args)
    #             push!(ordered_args, remaining[i])
    #             remaining[i] = nothing
    #             continue
    #         end
    #         if all(isnothing.(remaining))
    #             run = false
    #         end

    #         term = remaining[i]
    #         term_indices = get_free_indices(term)

    #         #if !isempty(intersect(term_indices, target_indices))
    #         # goes at beginning or end

    #         for fixed ∈ (ordered_args[1], ordered_args[end])
    #             fixed_indices = get_free_indices(fixed)

    #             @show fixed

    #             # next = nothing
    #             # next_indices = []
    #             # if j < length(ordered_args)
    #             #     next = ordered_args[j+1]
    #             #     next_indices = get_free_indices(next_indices)
    #             # end

    #             if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[1]
    #                 @show "adding 1" term
    #                 pushfirst!(ordered_args, term)
    #                 remaining[i] = nothing
    #                 break
    #             end

    #             if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[end]
    #                 @show "adding 2" term
    #                 push!(ordered_args, term)
    #                 remaining[i] = nothing
    #                 break
    #             end

    #             if  flip(term_indices[end]) == fixed_indices[end]
    #                 @show "adding 3" term
    #                 if typeof(fixed_indices[end]) == Lower
    #                     push!(ordered_args, term)
    #                 else
    #                     pushfirst!(ordered_args, term)
    #                 end
    #                 remaining[i] = nothing
    #                 break
    #             end

    #             if flip(term_indices[1]) == fixed_indices[1]
    #                 @show "adding 4" term
    #                 if typeof(fixed_indices[1]) == Lower
    #                     push!(ordered_args, term)
    #                 else
    #                     pushfirst!(ordered_args, term)
    #                 end
    #                 remaining[i] = nothing
    #                 break
    #             end
    #         end
                    
            
    #     end
    # # end

    # while run
        for i ∈ eachindex(remaining)
            if isnothing(remaining[i])
                continue
            end
            if isempty(ordered_args)
                term = remaining[i]
                ids = get_free_indices(term)
                if length(ids) == 2
                    if ids[1] ∉ target_indices && ids[2] ∈ target_indices
                        if typeof(ids[1]) == Upper
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), false))
                        elseif typeof(ids[1]) == Lower
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), true))
                        end
                    elseif ids[1] ∈ target_indices && ids[2] ∉ target_indices
                        if typeof(ids[2]) == Upper
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), true))
                        elseif typeof(ids[2]) == Lower
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), false))
                        end
                    elseif ids[1] ∉ target_indices && ids[2] ∉ target_indices
                        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), false))
                        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), true))
                        else
                            throw_not_std()
                        end
                    elseif ids[1] ∈ target_indices && ids[2] ∈ target_indices
                        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), false))
                        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter), Lower(ids[2].letter)), true))
                        else
                            throw_not_std()
                        end
                    end
                elseif length(ids) == 1
                    if ids[1] ∉ target_indices
                        if typeof(ids[1]) == Upper
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter)), false))
                        elseif typeof(ids[1]) == Lower
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter)), true))
                        end
                    elseif ids[1] ∈ target_indices # WETWET, can be rotated later
                        if typeof(ids[1]) == Upper
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter)), false))
                        elseif typeof(ids[1]) == Lower
                            push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(ids[1].letter)), true))
                        end
                    end
                else
                    throw_not_std()
                end
                remaining[i] = nothing
                continue
            end
            if all(isnothing.(remaining))
                run = false
            end

            term = remaining[i]
            term_indices = get_free_indices(term)

            #if !isempty(intersect(term_indices, target_indices))
            # goes at beginning or end

            for fixed ∈ (ordered_args[1], ordered_args[end])
                fixed_indices = get_free_indices(fixed)

                # @show fixed
                # @show term

                # next = nothing
                # next_indices = []
                # if j < length(ordered_args)
                #     next = ordered_args[j+1]
                #     next_indices = get_free_indices(next_indices)
                # end

                if typeof(term_indices[end]) == Lower && flip(term_indices[end]) == fixed_indices[1]
                    # @show "adding 1" term
                    if length(term_indices) == 2
                        pushfirst!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter), Lower(term_indices[2].letter)), false))
                    elseif length(term_indices) == 1
                        pushfirst!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter)), true))
                    else
                        throw_not_std()
                    end
                    remaining[i] = nothing
                    break
                end

                if typeof(term_indices[1]) == Upper && flip(term_indices[1]) == fixed_indices[end]
                    # @show "adding 2" term
                    if length(term_indices) == 2
                        push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter), Lower(term_indices[2].letter)), false))
                    elseif length(term_indices) == 1
                        push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter)), false))
                    else
                        throw_not_std()
                    end
                    remaining[i] = nothing
                    break
                end

                if flip(term_indices[end]) == fixed_indices[end] && typeof(term_indices[end]) == Upper
                    # @show "adding 3" term
                    if length(term_indices) == 2
                        push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter), Lower(term_indices[2].letter)), true))
                    elseif length(term_indices) == 1
                        pushfirst!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter)), false))
                    else
                        throw_not_std()
                    end
                    remaining[i] = nothing
                    break
                end

                if flip(term_indices[end]) == fixed_indices[end] && typeof(term_indices[end]) == Lower
                    # @show "adding 3" term
                    if length(term_indices) == 2
                        pushfirst!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter), Lower(term_indices[2].letter)), false))
                    elseif length(term_indices) == 1
                        pushfirst!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter)), true))
                    else
                        throw_not_std()
                    end
                    remaining[i] = nothing
                    break
                end

                if flip(term_indices[1]) == fixed_indices[1] && typeof(term_indices[1]) == Upper
                    # @show "adding 4" term
                    if length(term_indices) == 2
                        push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter), Lower(term_indices[2].letter)), false))
                    elseif length(term_indices) == 1
                        push!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter)), false))
                    else
                        throw_not_std()
                    end
                    remaining[i] = nothing
                    break
                end

                if flip(term_indices[1]) == fixed_indices[1] && typeof(term_indices[1]) == False
                    # @show "adding 4" term
                    if length(term_indices) == 2
                        pushfirst!(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter), Lower(term_indices[2].letter)), true))
                    elseif length(term_indices) == 1
                        push(ordered_args, StdizedTerm(Tensor(term.id, Upper(term_indices[1].letter)), true))
                    else
                        throw_not_std()
                    end
                    remaining[i] = nothing
                    break
                end
            end
                    
            
        end
    # end

    standardized_term = nothing

    for t ∈ ordered_args
        term = t.transpose ? evaluate(t.term') : t.term

        if isnothing(standardized_term)
            standardized_term = term
            continue
        end

        standardized_term = standardized_term * term
    end

    # @show standardized_term

    ordered_expr_ids = get_free_indices(evaluate(standardized_term))

    @assert length(ordered_expr_ids) == length(target_indices)

    # TODO: Should check that all indices are compliant with the target after rotation
    if ordered_expr_ids[end] == target_indices[end]
        # all is well
    elseif flip(ordered_expr_ids[1]) == target_indices[end]
        ordered_args = transpose_sequence(ordered_args)
    else
        throw_not_std()
    end

    argstr = ""

    # @show ordered_args
    
    for t ∈ ordered_args
        argstr *= to_std_string(t.term, t.transpose)
    end

    # INSTEAD: pop one at the time and add transpose as needed based on the index covariances
    # for (l,r) ∈ zip(ordered_args[1:end-1], ordered_args[2:end])
    #     l_indices = get_free_indices(l)
    #     r_indices = get_free_indices(r)

    #     @show l r

    #     if typeof(l_indices[end]) == Lower
    #         if length(l_indices) == 2
    #             argstr *= to_std_string(l, false)
    #         elseif length(l_indices) == 1
    #             argstr *= to_std_string(l, true)
    #         else
    #             throw_not_std()
    #         end
    #     else
    #         argstr *= to_std_string(l, true)
    #     end
    # end

    # argstr *= to_std_string(ordered_args[end])

    # must_t = false

    # if length(target_indices) == 2
    #     if typeof(target_indices[1]) == Upper && typeof(target_indices[2]) == Lower || typeof(target_indices[1]) == Lower && typeof(target_indices[2]) == Upper
    #         has_row_vector = false
    #         for term ∈ ordered_args
    #             ids = get_free_indices(term)
    #             if length(ids) == 1 && typeof(ids[1]) == Lower
    #                 has_row_vector = true
    #             end
    #         end

    #         if has_row_vector
    #             ordered_args = collect(reverse(ordered_args))
    #             must_t = true
    #         end
    #     end
    # elseif length(target_indices) == 1
    #     if typeof(target_indices[1]) == Upper
    #         has_row_vector = false
    #         for term ∈ ordered_args
    #             ids = get_free_indices(term)
    #             if length(ids) == 1 && typeof(ids[1]) == Lower
    #                 has_row_vector = true
    #             end
    #         end

    #         if has_row_vector
    #             ordered_args = collect(reverse(ordered_args))
    #             must_t = true
    #         end
    #     elseif typeof(target_indices[1]) == Lower
    #         has_col_vector = false
    #         for term ∈ ordered_args
    #             ids = get_free_indices(term)
    #             if length(ids) == 1 && typeof(ids[1]) == Upper
    #                 has_col_vector = true
    #             end
    #         end

    #         if has_col_vector
    #             ordered_args = collect(reverse(ordered_args))
    #             must_t = true
    #         end
    #     end
    # else
    #     throw_not_std()
    # end

    # @show ordered_args

    # for t ∈ ordered_args
    #     indices = get_free_indices(t)

    #     if typeof(indices[end]) == Lower
    #         if length(indices) == 2
    #             argstr *= to_std_string(t, transpose)
    #         elseif length(indices) == 1
    #             argstr *= to_std_string(t, !transpose)
    #         else
    #             throw_not_std()
    #         end
    #     else
    #         argstr *= to_std_string(t, transpose)
    #     end
    # end

    # if !isempty(target_indices)
    #     if typeof(target_indices[1]) == Lower
    #         argstr = "(" * argstr * ")ᵀ"
    #     end
    # end

    if trace
        argstr = "tr(" * argstr * ")"
    end

    # return ordered_args
    return argstr
end


# function to_std_string(arg::BinaryOperation{Mult})
#     arg1_ids = get_free_indices(arg.arg1)
#     arg2_ids = get_free_indices(arg.arg2)
#     arg_ids = get_free_indices(arg)

#     if arg.arg1 isa Real || arg.arg2 isa Real
#         return to_std_string(arg.arg1) * to_std_string(arg.arg2)
#     end

#     if !can_contract(arg.arg1, arg.arg2)
#         if arg1_ids == arg2_ids
#             return to_std_string(arg.arg1) * " ⊙ " * to_std_string(arg.arg2)
#         end

#         if length(arg1_ids) == 1 && length(arg2_ids) == 1
#             if typeof(arg1_ids[1]) == Upper && typeof(arg2_ids[1]) == Lower
#                 return to_std_string(arg.arg1) * to_std_string(arg.arg2)
#             end

#             if typeof(arg1_ids[1]) == Lower && typeof(arg2_ids[1]) == Upper
#                 return to_std_string(arg.arg2) * to_std_string(arg.arg1)
#             end
#         end

#         if length(arg1_ids) > 2 || length(arg2_ids) > 2 # is an outer product
#             throw_not_std()
#         end

#         return to_std_string(arg.arg1) * to_std_string(arg.arg2)
#     end

#     if isempty(arg_ids) # The result is a scalar
#         if length(arg1_ids) == 1 && length(arg2_ids) == 1
#             if typeof(arg1_ids[1]) == Lower
#                 return to_std_string(arg.arg1) * to_std_string(arg.arg2)
#             else # if typeof(arg1_ids[1]) == Upper
#                 return to_std_string(arg.arg2) * to_std_string(arg.arg1)
#             end
#         end

#         throw_not_std()
#     elseif length(arg_ids) == 1 # The result is a vector
#         arg1 = arg.arg1
#         arg2 = arg.arg2

#         if length(arg1_ids) == 1 && length(arg2_ids) == 2
#             arg1_ids, arg2_ids = arg2_ids, arg1_ids
#             arg1, arg2 = arg2, arg1
#         end

#         # arg1 is 2d matrix, arg2 is 1d

#         if typeof(arg_ids[1]) == Lower
#             if typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Lower
#                 if flip(arg1_ids[1]) == arg2_ids[1]
#                     return parenthesize_std(arg2) * parenthesize_std(arg1)
#                 else
#                     return parenthesize_std(arg2) * "ᵀ" * parenthesize_std(arg1) * "ᵀ"
#                 end
#             elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Upper
#                 if flip(arg1_ids[1]) == arg2_ids[1]
#                     return parenthesize_std(arg1) * "ᵀ" * parenthesize_std(arg2) * "ᵀ"
#                 else
#                     return parenthesize_std(arg2) * parenthesize_std(arg1)
#                 end
#             elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Lower
#                 if typeof(arg1) != Tensor
#                     throw_not_std()
#                 end

#                 if flip(arg1_ids[end]) == arg2_ids[1]
#                     return parenthesize_std(arg2) * "ᵀ" * get_sym(arg1) * "ᵀ"
#                 else
#                     return parenthesize_std(arg2) * "ᵀ" * get_sym(arg1)
#                 end
#             end
#         else # typeof(arg_ids[1]) == Upper
#             if typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Lower
#                 if flip(arg1_ids[1]) == arg2_ids[1]
#                     return parenthesize_std(arg2) * "ᵀ" * parenthesize_std(arg1)
#                 else
#                     return parenthesize_std(arg1) * parenthesize_std(arg2)
#                 end
#             elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Upper
#                 if flip(arg1_ids[2]) == arg2_ids[1]
#                     return parenthesize_std(arg1) * parenthesize_std(arg2)
#                 else
#                     return parenthesize_std(arg1) * parenthesize_std(arg2)
#                 end
#             elseif typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Upper
#                 if typeof(arg1) != Tensor || typeof(arg2) != Tensor
#                     throw_not_std()
#                 end

#                 if flip(arg1_ids[end]) == arg2_ids[1]
#                     return get_sym(arg1) * get_sym(arg2)
#                 else
#                     return get_sym(arg1) * "ᵀ" * get_sym(arg2)
#                 end
#             end
#         end
#     elseif length(arg_ids) == 2 # The result is a matrix
#         if length(arg1_ids) == 2 && length(arg2_ids) == 2
#             if flip(arg1_ids[end]) == arg2_ids[1]
#                 if typeof(arg1_ids[end]) == Lower
#                     return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
#                 else
#                     return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
#                 end
#             elseif flip(arg1_ids[1]) == arg2_ids[1]
#                 if typeof(arg1_ids[1]) == Lower
#                     return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
#                 else
#                     return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
#                 end
#             elseif flip(arg1_ids[end]) == arg2_ids[end]
#                 if typeof(arg1_ids[end]) == Lower
#                     return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
#                 else
#                     return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
#                 end
#             elseif flip(arg1_ids[1]) == arg2_ids[end]
#                 if typeof(arg1_ids[1]) == Lower
#                     return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
#                 else
#                     return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
#                 end
#             end
#         end
#     end

#     throw_not_std()
# end
