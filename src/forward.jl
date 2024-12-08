export evaluate

function diff(arg::Tensor, wrt::Tensor)
    if arg.id == wrt.id
        @assert length(arg.indices) == length(wrt.indices)

        D = 1

        for (u, l) ∈ zip(arg.indices, wrt.indices)
            D = BinaryOperation{*}(D, KrD(u, flip(l)))
        end

        return evaluate(D) # evaluate to get rid of the constant factor
    end

    return Zero(arg.indices..., [flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::KrD, wrt::Tensor)
    # This is arguably inconsistent with Tensor but the result will be Zero anyway
    # and this way the double indices are confined to the KrDs.
    return BinaryOperation{*}(arg, Zero([flip(i) for i ∈ wrt.indices]...))
end

function diff(arg::Real, wrt::Tensor)
    return Zero([flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::Adjoint, wrt::Tensor)
    return diff(arg.expr, wrt)
end

function diff(arg::Negate, wrt::Tensor)
    return Negate(diff(arg.arg, wrt))
end

function diff(arg::Sin, wrt::Tensor)
    return BinaryOperation{*}(Cos(arg.arg), diff(arg.arg, wrt))
end

function diff(arg::Cos, wrt::Tensor)
    return BinaryOperation{*}(Negate(Sin(arg.arg)), diff(arg.arg, wrt))
end

function diff(arg::BinaryOperation{*}, wrt::Tensor)
    return BinaryOperation{+}(
        BinaryOperation{*}(arg.arg1, diff(arg.arg2, wrt)),
        BinaryOperation{*}(diff(arg.arg1, wrt), arg.arg2),
    )
end

function diff(arg::BinaryOperation{Op}, wrt::Tensor) where {Op}
    return BinaryOperation{Op}(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function evaluate(arg::Negate)
    return Negate(evaluate(arg.arg))
end

function evaluate(arg::Adjoint)
    return evaluate(arg.expr) # The decorator is only needed when creating contractions
end

function evaluate(arg::Union{Tensor,KrD,Zero,Real,Diag})
    arg
end

function evaluate(arg::Sin)
    return Sin(evaluate(arg.arg))
end

function evaluate(arg::Cos)
    return Cos(evaluate(arg.arg))
end

function evaluate(::typeof(*), arg1::BinaryOperation{*}, arg2::Real)
    return evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::Real, arg2::BinaryOperation{*})
    if arg2.arg1 isa Real
        return BinaryOperation{*}(arg1 * arg2.arg1, arg2.arg2)
    elseif arg2.arg2 isa Real
        return BinaryOperation{*}(arg1 * arg2.arg2, arg2.arg1)
    else
        return BinaryOperation{*}(arg1, evaluate(arg2))
    end
end

function is_elementwise_multiplication(arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return !isempty(intersect(arg1_indices, arg2_indices))
end

function evaluate(::typeof(*), arg1::Tensor, arg2::BinaryOperation{*})
    is_elementwise =  is_elementwise_multiplication(arg2.arg1, arg2.arg2)

    if can_contract(arg1, arg2.arg1) && !is_elementwise
        new_arg1 = evaluate(*, arg1, arg2.arg1)
        return BinaryOperation{*}(new_arg1, arg2.arg2)
    elseif can_contract(arg1, arg2.arg2) && !is_elementwise
        new_arg2 = evaluate(*, arg1, arg2.arg2)
        return BinaryOperation{*}(arg2.arg1, new_arg2)
    else
        return BinaryOperation{*}(arg1, arg2)
    end
end

function evaluate(::typeof(*), arg1::BinaryOperation{*}, arg2::Tensor)
    is_elementwise =  is_elementwise_multiplication(arg1.arg1, arg1.arg2)

    if can_contract(arg1.arg2, arg2) && !is_elementwise
        new_arg2 = evaluate(*, arg1.arg2, arg2)
        return BinaryOperation{*}(arg1.arg1, new_arg2)
    elseif can_contract(arg1.arg1, arg2) && !is_elementwise
        new_arg1 = evaluate(*, arg1.arg1, arg2)
        return BinaryOperation{*}(new_arg1, arg1.arg2)
    else
        return BinaryOperation{*}(arg1, arg2)
    end
end

function is_trace(arg1::TensorValue, arg2::KrD)
    arg1_indices = get_free_indices(arg1)

    if flip(arg2.indices[1]) ∈ arg1_indices && flip(arg2.indices[2]) ∈ arg1_indices
        return true
    end

    return false
end

function evaluate(::typeof(*), arg1::KrD, arg2::BinaryOperation{*})
    if is_elementwise_multiplication(arg2.arg1, arg2.arg2) && can_contract(arg1, arg2.arg1) && can_contract(arg1, arg2.arg2)
        return BinaryOperation{*}(
            evaluate(*, arg1, arg2.arg1),
            evaluate(*, arg1, arg2.arg2),
        )
    elseif can_contract(arg1, arg2.arg1)
        new_arg1 = evaluate(*, arg1, arg2.arg1)
        return BinaryOperation{*}(new_arg1, evaluate(arg2.arg2))
    elseif can_contract(arg1, arg2.arg2)
        new_arg2 = evaluate(*, arg1, arg2.arg2)
        return BinaryOperation{*}(evaluate(arg2.arg1), new_arg2)
    else
        return BinaryOperation{*}(arg1, evaluate(arg2))
    end
end

function evaluate(::typeof(*), arg1::BinaryOperation{*}, arg2::KrD)
    if is_elementwise_multiplication(arg1.arg1, arg1.arg2) && can_contract(arg1.arg1, arg2) && can_contract(arg1.arg2, arg2)
        return BinaryOperation{*}(
            evaluate(*, arg1.arg1, arg2),
            evaluate(*, arg1.arg2, arg2),
        )
    elseif can_contract(arg1.arg2, arg2)
        new_arg2 = evaluate(*, arg1.arg2, arg2)
        return BinaryOperation{*}(evaluate(arg1.arg1), new_arg2)
    elseif can_contract(arg1.arg1, arg2)
        new_arg1 = evaluate(*, arg1.arg1, arg2)
        return BinaryOperation{*}(new_arg1, evaluate(arg1.arg2))
    else
        return BinaryOperation{*}(arg1, arg2)
    end
end

# TODO
# function evaluate(::typeof(*), arg1::KrD, arg2::Diag)

function evaluate(::typeof(*), arg1::Diag, arg2::KrD)
    @assert !is_trace(arg1, arg2) "Not implemented"

    @assert can_contract(arg1, arg2)
    @assert length(arg2.indices) == 2

    newarg = deepcopy(arg1)
    empty!(newarg.indices)

    contracted = false

    for i ∈ arg1.indices
        if flip(i) == arg2.indices[1] && !contracted
            push!(newarg.indices, arg2.indices[2])
            contracted = true
        elseif flip(i) == arg2.indices[2] && !contracted
            push!(newarg.indices, arg2.indices[1])
            contracted = true
        else
            push!(newarg.indices, i)
        end
    end

    return newarg
end

# TODO
# function evaluate(::typeof(*), arg1::Tensor, arg2::Diag)

function evaluate(::typeof(*), arg1::Diag, arg2::Tensor)
    @assert can_contract(arg1, arg2)

    arg1_inner_indices = get_free_indices(arg1.arg)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert length(arg1_indices) == 2
    @assert length(arg1_inner_indices) == 1

    if length(arg2_indices) != 1
        return BinaryOperation{*}(arg1, arg2)
    end

    @assert flip(arg2_indices[1]) ∈ arg1_indices

    new_index = if flip(arg2_indices[1]) == arg1_indices[1]
            arg1_indices[2]
        else
            @assert flip(arg2_indices[1]) == arg1_indices[2]
            arg1_indices[1]
        end

    new_arg1 = evaluate(reshape(arg1.arg, arg1_inner_indices[1], new_index))
    new_arg2 = evaluate(reshape(arg2, arg2_indices[1], new_index))

    return BinaryOperation{*}(new_arg1, new_arg2)
end

function evaluate(::typeof(*), arg1::Zero, arg2::TensorValue)
    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(union(arg1_free_indices, arg2_free_indices)...)
end

function evaluate(::typeof(*), arg1::TensorValue, arg2::Zero)
    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(union(arg1_free_indices, arg2_free_indices)...)
end

function evaluate(::typeof(*), arg1::KrD, arg2::Zero)
    contracting_index = eliminated_indices(get_free_indices(arg1), get_free_indices(arg2))

    if isempty(contracting_index)
        return Zero(union(arg1.indices, arg2.indices)...)
    end

    @assert can_contract(arg1, arg2)
    @assert length(arg2.indices) == 2

    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(union(arg1_free_indices, arg2_free_indices)...)
end

function evaluate(::typeof(*), arg1::Zero, arg2::KrD)
    contracting_index = eliminated_indices(get_free_indices(arg1), get_free_indices(arg2))

    if isempty(contracting_index)
        return Zero(union(arg1.indices, arg2.indices)...)
    end

    @assert can_contract(arg1, arg2)
    @assert length(arg2.indices) == 2

    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(union(arg1_free_indices, arg2_free_indices)...)
end

function evaluate(::typeof(*), arg1::KrD, arg2::Tensor)
    arg2_indices = get_free_indices(arg2)
    contracting_index = eliminated_indices(get_free_indices(arg1), arg2_indices)

    @assert length(eliminate_indices(get_free_indices(arg1), arg2_indices)) >= 1

    if is_trace(arg2, arg1)
        return BinaryOperation{*}(arg1, arg2)
    end

    if length(arg2_indices) == 1 && arg2_indices[1] ∈ arg1.indices
        return diag(arg2, arg1.indices...)
    end

    if isempty(contracting_index) # Is an outer product
        return BinaryOperation{*}(arg1, arg2)
    end

    @assert can_contract(arg1, arg2)
    @assert length(arg1.indices) == 2

    newarg = deepcopy(arg2)
    empty!(newarg.indices)

    contracted = false

    for i ∈ arg2.indices
        if flip(i) == arg1.indices[1] && !contracted
            push!(newarg.indices, arg1.indices[2])
            contracted = true
        elseif flip(i) == arg1.indices[2] && !contracted
            push!(newarg.indices, arg1.indices[1])
            contracted = true
        else
            push!(newarg.indices, i)
        end
    end

    newarg
end

function evaluate(::typeof(*), arg1::Union{Tensor,KrD}, arg2::KrD)
    arg1_indices = get_free_indices(arg1)
    contracting_index = eliminated_indices(arg1_indices, get_free_indices(arg2))

    @assert length(eliminate_indices(arg1_indices, get_free_indices(arg2))) >= 1

    if isempty(contracting_index) # Is an outer product
        return BinaryOperation{*}(arg1, arg2)
    end

    if is_trace(arg1, arg2)
        return BinaryOperation{*}(arg1, arg2)
    end

    if length(arg1_indices) == 1 && arg1_indices[1] ∈ arg2.indices
        return diag(arg1, arg2.indices...)
    end

    @assert can_contract(arg1, arg2)
    @assert length(arg2.indices) == 2

    newarg = deepcopy(arg1)
    empty!(newarg.indices)

    contracted = false

    for i ∈ arg1.indices
        if flip(i) == arg2.indices[1] && !contracted
            push!(newarg.indices, arg2.indices[2])
            contracted = true
        elseif flip(i) == arg2.indices[2] && !contracted
            push!(newarg.indices, arg2.indices[1])
            contracted = true
        else
            push!(newarg.indices, i)
        end
    end

    newarg
end

function evaluate(::typeof(*), arg1::BinaryOperation{+}, arg2::KrD)
    return evaluate(
        +,
        evaluate(*, evaluate(arg1.arg1), arg2),
        evaluate(*, evaluate(arg1.arg2), arg2),
    )
end

function evaluate(::typeof(*), arg1::KrD, arg2::BinaryOperation{+})
    return evaluate(
        +,
        evaluate(*, arg1, evaluate(arg2.arg1)),
        evaluate(*, arg1, evaluate(arg2.arg2)),
    )
end

function evaluate(::typeof(*), arg1::BinaryOperation{-}, arg2::KrD)
    return evaluate(
        -,
        evaluate(*, evaluate(arg1.arg1), arg2),
        evaluate(*, evaluate(arg1.arg2), arg2),
    )
end

function evaluate(::typeof(*), arg1::KrD, arg2::BinaryOperation{-})
    return evaluate(
        -,
        evaluate(*, arg1, evaluate(arg2.arg1)),
        evaluate(*, arg1, evaluate(arg2.arg2)),
    )
end

function evaluate(::typeof(*), arg1, arg2)
    return BinaryOperation{*}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::typeof(*), arg1::TensorValue, arg2::Real)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::Real, arg2::TensorValue)
    if arg1 == 1
        return arg2
    else
        BinaryOperation{*}(arg1, arg2)
    end
end

function evaluate(::typeof(*), arg1::Zero, arg2::Zero)
    new_indices = eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(new_indices...)
end

function evaluate(::typeof(+), arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    arg1
end

function evaluate(::typeof(+), arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg2)
end

function evaluate(::typeof(+), arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg1)
end

function evaluate(::typeof(+), arg1::Real, arg2::Real)
    return arg1 + arg2
end

function evaluate(::typeof(+), arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return BinaryOperation{*}(2, arg1)
    end

    return BinaryOperation{+}(arg1, arg2)
end

function evaluate(::typeof(+), arg1::NonTrivialNonMult, arg2::BinaryOperation{*})
    return evaluate(+, evaluate(arg2), evaluate(arg1))
end

function evaluate(::typeof(+), arg1::BinaryOperation{*}, arg2::NonTrivialValue)
    if evaluate(arg1) == evaluate(arg2)
        return BinaryOperation{*}(2, evaluate(arg1))
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{*}(evaluate(arg1.arg1) + 1, evaluate(arg2))
    end

    if evaluate(arg1.arg2) isa Real && evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{*}(evaluate(arg1.arg2) + 1, evaluate(arg2))
    end

    return BinaryOperation{+}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::typeof(-), arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    arg1
end

function evaluate(::typeof(-), arg1::Zero, arg2)
    @assert is_permutation(arg1, arg2)

    return Negate(evaluate(arg2))
end

function evaluate(::typeof(-), arg1, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg1)
end

function evaluate(::typeof(-), arg1::Real, arg2::Real)
    return arg1 - arg2
end

function evaluate(::typeof(-), arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return Zero(arg1_indices...)
    end

    return BinaryOperation{-}(arg1, arg2)
end

function evaluate(::typeof(-), arg1::NonTrivialNonMult, arg2::BinaryOperation{*})
    return evaluate(+, evaluate(Negate(arg2)), evaluate(arg1))
end

function evaluate(::typeof(-), arg1::BinaryOperation{*}, arg2::NonTrivialValue)
    if evaluate(arg1) == evaluate(arg2)
        arg1_indices = get_free_indices(arg1)
        return Zero(arg1_indices...)
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{*}(evaluate(arg1.arg1) - 1, evaluate(arg2))
    end

    if evaluate(arg1.arg2) isa Real && evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{*}(evaluate(arg1.arg2) - 1, evaluate(arg2))
    end

    return BinaryOperation{-}(evaluate(arg1), evaluate(arg2))
end

function evaluate(op::BinaryOperation{*})
    evaluate(*, evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(op::BinaryOperation{Op}) where {Op}
    evaluate(Op, evaluate(op.arg1), evaluate(op.arg2))
end
