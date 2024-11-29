export create_matrix
export create_vector

export differential

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
    if length(arg.indices) == 2
        if typeof(arg.indices[1]) == Upper && typeof(arg.indices[2]) == Lower
            return arg.id
        end

        if typeof(arg.indices[1]) == Lower && typeof(arg.indices[2]) == Upper
            return arg.id * "ᵀ"
        end
    end

    if length(arg.indices) == 1
        if typeof(arg.indices[1]) == Upper
            return arg.id
        end

        if typeof(arg.indices[1]) == Lower
            return arg.id * "ᵀ"
        end
    end

    throw_not_std()
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

function get_sym(arg::Tensor)
    return arg.id
end

function throw_not_std()
    throw(DomainError("Cannot write expression in standard notation"))
end

function to_std_string(arg::BinaryOperation{+})
    @assert are_indices_equivalent(arg.arg1, arg.arg2)

    return to_std_string(arg.arg1) * " + " * to_std_string(arg.arg2)
end

function to_std_string(arg::BinaryOperation{*})
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)
    arg_ids = get_free_indices(arg)

    if !can_contract(arg.arg1, arg.arg2)
        return _to_std_string(arg.arg1) * _to_std_string(arg.arg2)
    end

    if length(arg_ids) == 1 # The result is a vector
        arg1 = arg.arg1
        arg2 = arg.arg2

        if length(arg1_ids) == 1 && length(arg2_ids) == 2
            arg1_ids, arg2_ids = arg2_ids, arg1_ids
            arg1, arg2 = arg2, arg1
        end

        # arg1 is 2d matrix, arg2 is 1d

        if typeof(arg_ids[1]) == Lower
            if typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Lower
                if flip(arg1_ids[1]) == arg2_ids[1]
                    return parenthesize_std(arg2) * parenthesize_std(arg1)
                else
                    return parenthesize_std(arg2) * "ᵀ" * parenthesize_std(arg1) * "ᵀ"
                end
            elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Upper
                if flip(arg1_ids[1]) == arg2_ids[1]
                    return parenthesize_std(arg1) * "ᵀ" * parenthesize_std(arg2) * "ᵀ"
                else
                    return parenthesize_std(arg2) * parenthesize_std(arg1)
                end
            elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Lower
                if typeof(arg1) != Tensor
                    throw_not_std()
                end

                if flip(arg1_ids[end]) == arg2_ids[1]
                    return parenthesize_std(arg2) * "ᵀ" * get_sym(arg1) * "ᵀ"
                else
                    return parenthesize_std(arg2) * "ᵀ" * get_sym(arg1)
                end
            end
        else # typeof(arg_ids[1]) == Upper
            if typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Lower
                if flip(arg1_ids[1]) == arg2_ids[1]
                    return parenthesize_std(arg2) * "ᵀ" * parenthesize_std(arg1)
                else
                    return parenthesize_std(arg1) * parenthesize_std(arg2)
                end
            elseif typeof(arg1.indices[1]) == Lower && typeof(arg1.indices[2]) == Upper
                if flip(arg1_ids[2]) == arg2_ids[1]
                    return parenthesize_std(arg1) * parenthesize_std(arg2)
                else
                    return parenthesize_std(arg1) * parenthesize_std(arg2)
                end
            elseif typeof(arg1.indices[1]) == Upper && typeof(arg1.indices[2]) == Upper
                if typeof(arg1) != Tensor || typeof(arg2) != Tensor
                    throw_not_std()
                end

                if flip(arg1_ids[end]) == arg2_ids[1]
                    return get_sym(arg1) * get_sym(arg2)
                else
                    return get_sym(arg1) * "ᵀ" * get_sym(arg2)
                end
            end
        end
    elseif length(arg_ids) == 2 # The result is a matrix
        if length(arg1_ids) == 2 && length(arg2_ids) == 2
            if flip(arg1_ids[end]) == arg2_ids[1]
                if typeof(arg1_ids[end]) == Lower
                    return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
                else
                    return  parenthesize_std(arg.arg2) *
                            parenthesize_std(arg.arg1)
                end
            elseif flip(arg1_ids[1]) == arg2_ids[1]
                if typeof(arg1_ids[1]) == Lower
                    return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
                else
                    return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
                end
            elseif flip(arg1_ids[end]) == arg2_ids[end]
                if typeof(arg1_ids[end]) == Lower
                    return parenthesize_std(arg.arg1) * parenthesize_std(arg.arg2)
                else
                    return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
                end
            elseif flip(arg1_ids[1]) == arg2_ids[end]
                if typeof(arg1_ids[1]) == Lower
                    return  parenthesize_std(arg.arg1) *
                            parenthesize_std(arg.arg2)
                else
                    return parenthesize_std(arg.arg2) * parenthesize_std(arg.arg1)
                end
            end
        end
    end

    throw_not_std()
end
    