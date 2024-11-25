export create_matrix
export create_vector

export differential

const global REGISTERED_SYMBOLS = Dict{String, Tensor}()

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

function differential(expr, wrt::String)
    ∂ = Tensor(wrt)

    if wrt ∈ keys(REGISTERED_SYMBOLS)
        for index ∈ REGISTERED_SYMBOLS[wrt].indices
            push!(∂.indices, same(index, get_next_letter()))
        end
    end

    linear_form = diff(expr, ∂)
    linear_form = evaluate(linear_form)

    linear_form
end

function to_std_string(arg::Tensor)
    return arg.id
end

function to_std_string(arg::Real)
    return to_string(arg)
end

function to_std_string(arg::UnaryOperation)
    @assert false, "Not implemented"
end

function parenthesize_std(arg)
    return to_std_string(arg)
end

function parenthesize_std(arg::BinaryOperation{+})
    return "(" * to_std_string(arg) * ")"
end

function throw_not_std()
    throw(DomainError(arg, "Cannot write expression in standard notation"))
end

function to_std_string(arg::BinaryOperation{+})
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert are_indices_equivalent(arg.arg1, arg.arg2)

    if length(arg1_ids) == 1
        return to_std_string(arg.arg1) * " " * string(+) * " " * to_std_string(arg.arg2)
    elseif length(arg2_ids) == 2
        if (typeof(arg1_ids[1]) == Upper && typeof(arg1_ids[2]) == Lower) &&
           (typeof(arg2_ids[1]) == Upper && typeof(arg2_ids[2]) == Lower)

            return to_std_string(arg.arg1) * " " * string(+) * " " * to_std_string(arg.arg2)

        elseif (typeof(arg1_ids[1]) == Upper && typeof(arg1_ids[2]) == Lower) &&
               (typeof(arg2_ids[1]) == Lower && typeof(arg2_ids[2]) == Upper)

            if arg1_ids[1] == arg2_ids[2] && arg1_ids[2] == arg2_ids[1]
                return to_std_string(arg.arg1) * " " * string(+) * " " * to_std_string(arg.arg2) * "ᵀ"
            else
                throw_not_std()
            end
        elseif (typeof(arg1_ids[1]) == Lower && typeof(arg1_ids[2]) == Upper) &&
               (typeof(arg2_ids[1]) == Upper && typeof(arg2_ids[2]) == Lower)

            if arg1_ids[1] == arg2_ids[2] && arg1_ids[2] == arg2_ids[1]
                return to_std_string(arg.arg1) * "ᵀ " * string(+) * " " * to_std_string(arg.arg2)
            else
                throw_not_std()
            end
        end
    else
        throw_not_std()
    end
end

# TODO: Refactor this method
function to_std_string(arg::BinaryOperation{*})
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)
    arg_ids = get_free_indices(arg)

    if !can_contract(arg.arg1, arg.arg2)
        return to_std_string(arg.arg1) * to_std_string(arg.arg2)
    else
        if length(arg_ids) <= 1 # The result is a vector or a scalar
            arg1 = arg.arg1
            arg2 = arg.arg2

            if length(arg1_ids) == 1 && length(arg2_ids) == 2
                arg1_ids, arg2_ids = arg2_ids, arg1_ids
                arg1, arg2 = arg2, arg1
            end

            # arg1 is 2d matrix, arg2 is 1d

            if (typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Lower) ||
               (typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Upper)
                if flip(arg1_ids[1]) == arg2_ids[1]
                    if typeof(arg1_ids[1]) == Upper
                        return to_std_string(arg2) * "ᵀ" * to_std_string(arg1)
                    else
                        return to_std_string(arg1) * "ᵀ" * to_std_string(arg2)
                    end
                end

                if flip(arg1_ids[end]) == arg2_ids[1]
                    if typeof(arg1_ids[end]) == Lower
                        return to_std_string(arg1) * "" * to_std_string(arg2)
                    else
                        return to_std_string(arg2) * "ᵀ" * to_std_string(arg1) * "ᵀ"
                    end
                end
            elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Lower
                if flip(arg1_ids[end]) == arg2_ids[1]
                    return to_std_string(arg2) * "ᵀ" * to_std_string(arg1) * "ᵀ"
                else
                    return to_std_string(arg2) * "ᵀ" * to_std_string(arg1)
                end
            else # typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Upper
                if flip(arg1_ids[end]) == arg2_ids[1]
                    return to_std_string(arg1) * to_std_string(arg2)
                else
                    return to_std_string(arg1) * "ᵀ" * to_std_string(arg2)
                end
            end

            throw_not_std()
        elseif length(arg_ids) == 2 # The result is a matrix
            if !(typeof(arg_ids[1]) == Upper && typeof(arg_ids[2]) == Lower) &&
               !(typeof(arg_ids[1]) == Lower && typeof(arg_ids[2]) == Upper)
               throw_not_std()
            end
            if length(arg1_ids) == 2 && length(arg2_ids) == 2
                if flip(arg1_ids[end]) == arg2_ids[1]
                    if typeof(arg1_ids[end]) == Lower
                        return to_std_string(arg.arg1) * to_std_string(arg.arg2)
                    else
                        return "(" * to_std_string(arg.arg2) * to_std_string(arg.arg1) * ")ᵀ"
                    end
                elseif flip(arg1_ids[1]) == arg2_ids[1]
                    if typeof(arg1_ids[1]) == Lower
                        return to_std_string(arg.arg1) * "ᵀ" * to_std_string(arg.arg2)
                    else
                        return to_std_string(arg.arg2) * "ᵀ" * to_std_string(arg.arg1)
                    end
                elseif flip(arg1_ids[end]) == arg2_ids[end]
                    if typeof(arg1_ids[end]) == Lower
                        return to_std_string(arg.arg1) * to_std_string(arg.arg2) * "ᵀ"
                    else
                        return to_std_string(arg.arg2) * to_std_string(arg.arg1) * "ᵀ"
                    end
                elseif flip(arg1_ids[1]) == arg2_ids[end]
                    if typeof(arg1_ids[1]) == Lower
                        return "(" * to_std_string(arg.arg1) * to_std_string(arg.arg2) * ")ᵀ"
                    else
                        return to_std_string(arg.arg2) * to_std_string(arg.arg1)
                    end
                end
            end

            throw_not_std()
        else
            throw_not_std()
        end
    end
end