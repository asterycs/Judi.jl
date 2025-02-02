# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function diff(arg::Tensor, wrt::Tensor)
    if arg.id == wrt.id
        @assert length(arg.indices) == length(wrt.indices)

        D = 1

        for (u, l) ∈ zip(arg.indices, wrt.indices)
            D = BinaryOperation{Mult}(D, KrD(u, flip(l)))
        end

        return evaluate(D) # evaluate to get rid of the constant factor
    end

    return Zero(eliminate_indices(arg.indices)..., [flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::KrD, wrt::Tensor)
    # This is arguably inconsistent with Tensor but the result will be Zero anyway
    # and this way the double indices are confined to the KrDs.
    return BinaryOperation{Mult}(arg, Zero([flip(i) for i ∈ wrt.indices]...))
end

function diff(arg::Real, wrt::Tensor)
    return Zero([flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::Negate, wrt::Tensor)
    return Negate(diff(arg.arg, wrt))
end

function diff(arg::Sin, wrt::Tensor)
    return BinaryOperation{Mult}(Cos(arg.arg), diff(arg.arg, wrt))
end

function diff(arg::Cos, wrt::Tensor)
    return BinaryOperation{Mult}(Negate(Sin(arg.arg)), diff(arg.arg, wrt))
end

function diff(arg::BinaryOperation{Mult}, wrt::Tensor)
    return BinaryOperation{Add}(
        BinaryOperation{Mult}(arg.arg1, diff(arg.arg2, wrt)),
        BinaryOperation{Mult}(diff(arg.arg1, wrt), arg.arg2),
    )
end

function diff(arg::BinaryOperation{Op}, wrt::Tensor) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function evaluate(arg::Negate)
    return Negate(evaluate(arg.arg))
end

function evaluate(arg::Union{Tensor,KrD,Zero,Real})
    return arg
end

function evaluate(arg::Sin)
    return Sin(evaluate(arg.arg))
end

function evaluate(arg::Cos)
    return Cos(evaluate(arg.arg))
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::Real)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Real, arg2::BinaryOperation{Mult})
    if arg2.arg1 isa Real
        return BinaryOperation{Mult}(arg1 * arg2.arg1, arg2.arg2)
    elseif arg2.arg2 isa Real
        return BinaryOperation{Mult}(arg1 * arg2.arg2, arg2.arg1)
    else
        return BinaryOperation{Mult}(arg1, evaluate(arg2))
    end
end

# TODO: Create separate type for elementwise products and delete this method
function is_elementwise_multiplication(arg1::TensorValue, arg2::TensorValue)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # TODO: Make symmetric and refactor
    if typeof(arg1) == BinaryOperation{Mult}
        if (typeof(arg1.arg1) == KrD || typeof(arg1.arg2) == KrD) &&
           is_elementwise_multiplication(arg1.arg1, arg1.arg2)
            return can_contract(arg1, arg2)
        end
    end

    return !isempty(intersect(arg1_indices, arg2_indices))
end

function is_elementwise_multiplication(arg1, arg2)
    return false
end

function evaluate(::Mult, arg1::Tensor, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::Tensor)
    is_elementwise = is_elementwise_multiplication(arg1.arg1, arg1.arg2)
    arg1_indices, arg2_indices = get_free_indices.((arg1, arg2))

    contracting_indices = eliminated_indices(arg1_indices, arg2_indices)

    if is_elementwise &&
       is_diag(arg1) &&
       !isempty(contracting_indices) &&
       length(arg2_indices) == 1
        new_index = setdiff(arg1_indices, contracting_indices)
        old_index = intersect(arg1_indices, contracting_indices)

        @assert length(new_index) == 1
        @assert length(old_index) == 1
        new_index = new_index[1]
        old_index = old_index[1]

        # Either arg1.arg1 OR arg1.arg2 is a KrD when is_diag is true
        left_tensor = typeof(arg1.arg1) == KrD ? arg1.arg2 : arg1.arg1

        arg1 = evaluate(
            update_index(left_tensor, old_index, new_index, allow_shape_change = true),
        )
        arg2 = evaluate(
            update_index(arg2, flip(old_index), new_index, allow_shape_change = true),
        )

        return BinaryOperation{Mult}(arg1, arg2)
    elseif can_contract(arg1.arg2, arg2) && !is_elementwise
        new_arg2 = evaluate(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(arg1.arg1, new_arg2)
    elseif can_contract(arg1.arg1, arg2) && !is_elementwise
        new_arg1 = evaluate(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, arg1.arg2)
    else
        return BinaryOperation{Mult}(arg1, arg2)
    end
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    new_args = []

    available1 = Any[arg1.arg1; arg1.arg2]
    available2 = Any[arg2.arg1; arg2.arg2]

    for i ∈ eachindex(available1)
        if isnothing(available1[i])
            continue
        end
        for j ∈ eachindex(available2)
            if isnothing(available2[j]) || isnothing(available1[i])
                continue
            end

            if can_contract(available1[i], available2[j])
                push!(new_args, evaluate(Mult(), available1[i], available2[j]))
                available1[i] = nothing
                available2[j] = nothing
            end
        end
    end

    for i ∈ available1
        if !isnothing(i)
            push!(new_args, i)
        end
    end

    for i ∈ available2
        if !isnothing(i)
            push!(new_args, i)
        end
    end

    new_arg = nothing

    for args ∈ Iterators.partition(new_args, 2)
        if length(args) == 1
            if isnothing(new_arg)
                return args[1]
            else
                return evaluate(BinaryOperation{Mult}(new_arg, args[1]))
            end
        end

        if isnothing(new_arg)
            new_arg = evaluate(BinaryOperation{Mult}(args[1], args[2]))
        else
            new_arg =
                evaluate(BinaryOperation{Mult}(new_arg, BinaryOperation{Mult}(args[1], args[2])))
        end
    end

    return new_arg
end

function evaluate(::Mult, arg1::KrD, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::KrD)
    if is_elementwise_multiplication(arg1.arg1, arg1.arg2) &&
       can_contract(arg1.arg1, arg2) &&
       can_contract(arg1.arg2, arg2)
        return BinaryOperation{Mult}(
            evaluate(Mult(), arg1.arg1, arg2),
            evaluate(Mult(), arg1.arg2, arg2),
        )
    elseif can_contract(arg1.arg2, arg2)
        new_arg2 = evaluate(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg1), new_arg2)
    elseif can_contract(arg1.arg1, arg2)
        new_arg1 = evaluate(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, evaluate(arg1.arg2))
    elseif arg1.arg1 isa Real
        return BinaryOperation{Mult}(arg1.arg1, BinaryOperation{Mult}(arg1.arg2, arg2))
    elseif arg1.arg2 isa Real
        return BinaryOperation{Mult}(arg1.arg2, BinaryOperation{Mult}(arg1.arg1, arg2))
    else
        return BinaryOperation{Mult}(arg1, arg2)
    end
end

function evaluate(::Mult, arg1::Zero, arg2::UnaryOperation)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::UnaryOperation, arg2::Zero)
    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(union(arg1_free_indices, arg2_free_indices)...)
end

function evaluate(::Mult, arg1::Zero, arg2::TensorValue)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::TensorValue, arg2::Zero)
    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg1), get_free_indices(arg2))

    return Zero(union(arg1_free_indices, arg2_free_indices)...)
end

function evaluate(::Mult, arg1::KrD, arg2::Zero)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Zero, arg2::KrD)
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

function evaluate(::Mult, arg1::KrD, arg2::Tensor)
    arg2_indices = get_free_indices(arg2)
    contracting_index = eliminated_indices(get_free_indices(arg1), arg2_indices)

    @assert length(eliminate_indices(get_free_indices(arg1), arg2_indices)) >= 1

    if is_diag(arg2, arg1)
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if isempty(contracting_index) # Is an outer product
        return BinaryOperation{Mult}(arg1, arg2)
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

    return newarg
end

function evaluate(::Mult, arg1::Negate, arg2::TensorValue)
    return Negate(evaluate(Mult(), arg1.arg, arg2))
end

function evaluate(::Mult, arg1::TensorValue, arg2::Negate)
    return Negate(evaluate(Mult(), arg1, arg2.arg))
end

function evaluate(::Mult, arg1::Negate, arg2::KrD)
    return invoke(evaluate, Tuple{Mult,UnaryOperation,KrD}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::KrD, arg2::Negate)
    return invoke(evaluate, Tuple{Mult,KrD,UnaryOperation}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::UnaryOperation, arg2::KrD)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::KrD, arg2::UnaryOp) where {UnaryOp<:UnaryOperation}
    if can_contract(evaluate(arg1), evaluate(arg2.arg))
        return UnaryOp(evaluate(Mult(), evaluate(arg1), evaluate(arg2.arg)))
    end

    return BinaryOperation{Mult}(evaluate(arg2), evaluate(arg1))
end

function is_diag(arg1::KrD, arg2::TensorValue)
    return is_diag(arg2, arg1)
end

function is_diag(arg1::TensorValue, arg2::KrD)
    arg1_indices, arg2_indices = get_free_indices.((arg1, arg2))

    return length(arg1_indices) == 1 && !isempty(intersect(arg1_indices, arg2_indices))
end

function is_diag(arg1::KrD, arg2::KrD)
    return false
end

function is_diag(arg::BinaryOperation{Mult})
    return is_diag(arg.arg1, arg.arg2)
end

function is_diag(arg1, arg2)
    return false
end

function evaluate(::Mult, arg1::Union{Tensor,KrD}, arg2::KrD)
    arg1_indices = get_free_indices(arg1)
    contracting_index = eliminated_indices(arg1_indices, get_free_indices(arg2))

    @assert length(eliminate_indices(arg1_indices, get_free_indices(arg2))) >= 1

    if isempty(contracting_index) # Is an outer product
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if is_diag(arg1, arg2)
        return BinaryOperation{Mult}(arg1, arg2)
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

    return newarg
end

function evaluate(
    ::Mult,
    arg1::BinaryOperation{Op},
    arg2::Union{Tensor,KrD},
) where {Op<:AdditiveOperation}
    return evaluate(
        Op(),
        evaluate(Mult(), evaluate(arg1.arg1), evaluate(arg2)),
        evaluate(Mult(), evaluate(arg1.arg2), evaluate(arg2)),
    )
end

function evaluate(
    ::Mult,
    arg1::Union{Tensor,KrD},
    arg2::BinaryOperation{Op},
) where {Op<:AdditiveOperation}
    return evaluate(
        Op(),
        evaluate(Mult(), arg1, evaluate(arg2.arg1)),
        evaluate(Mult(), arg1, evaluate(arg2.arg2)),
    )
end

function evaluate(::Mult, arg1::Value, arg2::Value)
    return BinaryOperation{Mult}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Mult, arg1::Negate, arg2::Negate)
    return evaluate(Mult(), arg1.arg, arg2.arg)
end

function evaluate(::Mult, arg1::Negate, arg2::Zero)
    return invoke(evaluate, Tuple{Mult,TensorValue,Zero}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::Zero, arg2::Negate)
    return invoke(evaluate, Tuple{Mult,Zero,TensorValue}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::TensorValue, arg2::Real)
    evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Real, arg2::TensorValue)
    if arg1 == 1
        return arg2
    else
        BinaryOperation{Mult}(arg1, arg2)
    end
end

function evaluate(::Mult, arg1::Zero, arg2::Zero)
    new_indices = eliminate_indices([get_free_indices(arg1); get_free_indices(arg2)])

    return Zero(new_indices...)
end

function evaluate(::Mult, arg1::Zero, arg2::Real)
    return evaluate(arg1)
end

function evaluate(::Mult, arg1::Real, arg2::Zero)
    return evaluate(arg2)
end

function evaluate(::Add, arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return arg1
end

function evaluate(::Add, arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg2)
end

function evaluate(::Add, arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg1)
end

function evaluate(::Add, arg1::Real, arg2::Real)
    return arg1 + arg2
end

function evaluate(::Add, arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return BinaryOperation{Mult}(2, arg1)
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::Zero)
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    # TODO: extend
    return _add_to_product(arg1, arg2)
end

function evaluate(::Add, arg1::UnaryValue, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::UnaryValue)
    return _add_to_product(arg1, arg2)
end

# function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Mult})
#     return _add_to_product(arg2, arg1)
# end

# function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Sub})
#     return _add_to_product(arg1, arg2)
# end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::UnaryValue)
    if evaluate(arg1) == evaluate(arg2)
        return BinaryOperation{Mult}(2, evaluate(arg1))
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg1) + 1, evaluate(arg2))
    end

    if evaluate(arg1.arg2) isa Real && evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg2) + 1, evaluate(arg2))
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Add})
    if evaluate(arg1) == evaluate(arg2.arg1)
        return BinaryOperation{Add}(BinaryOperation{Mult}(2, evaluate(arg1)), evaluate(arg2.arg2))
    end

    if evaluate(arg1) == evaluate(arg2.arg2)
        return BinaryOperation{Add}(BinaryOperation{Mult}(2, evaluate(arg1)), evaluate(arg2.arg1))
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Sub})
    if evaluate(arg1) == evaluate(arg2.arg1)
        return BinaryOperation{Sub}(BinaryOperation{Mult}(2, evaluate(arg1)), evaluate(arg2.arg2))
    end

    if evaluate(arg1) == evaluate(arg2.arg2)
        return evaluate(arg2.arg1)
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if evaluate(arg1) == evaluate(arg2)
        return BinaryOperation{Mult}(2, evaluate(arg1))
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function evaluate(
    ::Add,
    arg1::BinaryOperation{Op},
    arg2::Zero,
) where {Op<:AdditiveOperation}
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Add})
    # TODO: extend and change the below overload to (BinaryOp{Add}, UnaryValue)
    return invoke(evaluate, Tuple{Add,BinaryOperation{Add},Value}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::Zero)
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::Zero, arg2::BinaryOperation{Add})
    return invoke(evaluate, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::Zero, arg2::BinaryOperation{Mult})
    invoke(evaluate, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::Value, arg2::BinaryOperation{Add})
    return evaluate(Add(), arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Add})
    # TODO: extend
    return _add_to_product(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::Value)
    if evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, evaluate(arg1.arg1)),
            evaluate(arg1.arg2),
        )
    end

    if evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, evaluate(arg1.arg2)),
            evaluate(arg1.arg1),
        )
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Add, arg1::Value, arg2::BinaryOperation{Sub})
    return evaluate(Add(), arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::Value)
    if evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{Sub}(
            BinaryOperation{Mult}(2, evaluate(arg1.arg1)),
            evaluate(arg1.arg2),
        )
    end

    if evaluate(arg1.arg2) == evaluate(arg2)
        return evaluate(arg1.arg1)
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::Zero)
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Sub})
    return evaluate(Add, arg2, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Add})
    if arg1.arg1 == arg2.arg1
        return evaluate(BinaryOperation{Add}(
            evaluate(BinaryOperation{Mult}(2, arg1.arg1)),
            evaluate(BinaryOperation{Sub}(arg2.arg2, arg1.arg2)),
        ))
    end

    if arg1.arg1 == arg2.arg2
        return evaluate(BinaryOperation{Add}(
            evaluate(BinaryOperation{Mult}(2, arg1.arg1)),
            evaluate(BinaryOperation{Sub}(arg2.arg1, arg1.arg2)),
        ))
    end

    if arg1.arg2 == arg2.arg1
        return BinaryOperation{Add}(arg1.arg1, arg2.arg2)
    end

    if arg1.arg2 == arg2.arg2
        return BinaryOperation{Add}(arg1.arg1, arg2.arg1)
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Sub})
    # TODO: extend
    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::Zero, arg2::BinaryOperation{Sub})
    return invoke(evaluate, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function evaluate(::Sub, arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return arg1
end

function evaluate(::Sub, arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return Negate(evaluate(arg2))
end

function evaluate(::Sub, arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg1)
end

function evaluate(::Sub, arg1::Real, arg2::Real)
    return arg1 - arg2
end

function evaluate(::Sub, arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return Zero(arg1_indices...)
    end

    return BinaryOperation{Sub}(arg1, arg2)
end

function evaluate(::Sub, arg1::Zero, arg2::BinaryOperation{Mult})
    return invole(evaluate, Tuple{Sub,Zero,Value}, Sub(), arg1, arg2)
end

function evaluate(::Sub, arg1::BinaryOperation{Mult}, arg2::Zero)
    return invole(evaluate, Tuple{Sub,Value,Zero}, Sub(), arg1, arg2)
end

function evaluate(::Sub, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    return _sub_from_product(arg1, arg2)
end

function evaluate(::Sub, arg1::Value, arg2::BinaryOperation{Mult})
    return _sub_from_product(arg2, arg1)
end

function evaluate(::Sub, arg1::BinaryOperation{Mult}, arg2::Value)
    return _sub_from_product(arg1, arg2)
end

function _sub_from_product(arg1::BinaryOperation{Mult}, arg2::Value)
    if evaluate(arg1) == evaluate(arg2)
        arg1_indices = get_free_indices(arg1)
        return Zero(arg1_indices...)
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg1) - 1, evaluate(arg2))
    end

    if evaluate(arg1.arg2) isa Real && evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg2) - 1, evaluate(arg2))
    end

    return BinaryOperation{Sub}(evaluate(arg1), evaluate(arg2))
end

function evaluate(op::BinaryOperation{Mult})
    evaluate(Mult(), evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(op::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    evaluate(Op(), evaluate(op.arg1), evaluate(op.arg2))
end
