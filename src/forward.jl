export evaluate

export D

function D(expr, wrt::Sym)
    linear_form = diff(expr, wrt)
    linear_form = evaluate(linear_form)

    linear_form
end

function diff(sym::Sym, wrt::Sym)
    if sym.id == wrt.id
        @assert length(sym.indices) == length(wrt.indices)

        D = nothing

        for (u,l) ∈ zip(sym.indices, wrt.indices)
            if isnothing(D)
                D = KrD(u, flip(l))
            else
                D = BinaryOperation{*}(D, KrD(u, flip(l)))
            end
        end

        return D
    else
        return Zero(sym.indices..., [flip(i) for i ∈ wrt.indices]...)
    end
end

function diff(arg::KrD, wrt::Sym)
    # This is arguably inconsistent with Sym but the result will be Zero anyway
    # and this way the double indices are confined to the KrDs.
    return BinaryOperation{*}(arg, Zero([flip(i) for i ∈ wrt.indices]...))
end

function diff(arg::UnaryOperation, wrt::Sym)
    UnaryOperation(arg.op, diff(arg.arg, wrt))
end

function diff(arg::BinaryOperation{*}, wrt::Sym)
    BinaryOperation{+}(BinaryOperation{*}(arg.arg1, diff(arg.arg2, wrt)), BinaryOperation{*}(diff(arg.arg1, wrt), arg.arg2))
end

function diff(arg::BinaryOperation{+}, wrt::Sym)
    BinaryOperation{+}(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function evaluate(sym::Adjoint)
    free_indices = get_free_indices(sym)

    e = sym.expr

    for i ∈ free_indices
        e = BinaryOperation{*}(e, KrD(i, i))
    end

    return evaluate(e)
end

function evaluate(sym::Sym)
    sym
end

function evaluate(arg::UnaryOperation)
    if typeof(arg.op) == KrD
        return evaluate(*, evaluate(arg.arg), arg.op)
    end

    return arg
end

function evaluate(::typeof(*), arg1::KrD, arg2::BinaryOperation{*})
    evaluate(*, evaluate(arg2), arg1)
end

function evaluate(::typeof(*), arg1::BinaryOperation{*}, arg2::KrD)
    if can_contract(arg1.arg2, arg2)
        new_arg2 = evaluate(*, arg1.arg2, arg2)
        return BinaryOperation{*}(arg1.arg1, new_arg2)
    elseif can_contract(arg1.arg1, arg2)
        new_arg1 = evaluate(*, arg1.arg1, arg2)
        return BinaryOperation{*}(new_arg1, arg1.arg2)
    else
        return BinaryOperation{*}(arg1, arg2)
    end
end

function evaluate(::typeof(*), arg1::KrD, arg2::Sym)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::Union{Sym, Zero, KrD}, arg2::KrD)
    contracting_index = eliminated_indices([get_free_indices(arg1); get_free_indices(arg2)])

    if isempty(contracting_index) # One arg is a scalar or the indices are incompatible.
        return UnaryOperation(arg2, arg1)
    end

    @assert length(contracting_index) == 2

    if !can_contract(arg1, arg2) # Happens if one of the arguments is self-contracting
        return UnaryOperation(arg2, arg1)
    end

    @assert length(arg2.indices) == 2

    newarg = deepcopy(arg1)
    empty!(newarg.indices)

    for i ∈ arg1.indices
        if flip(i) == arg2.indices[1]
            push!(newarg.indices, arg2.indices[2])
        elseif flip(i) == arg2.indices[2]
            push!(newarg.indices, arg2.indices[1])
        else
            push!(newarg.indices, i)
        end
    end

    newarg
end

function evaluate(::typeof(*), arg1, arg2)
    return BinaryOperation{*}(evaluate(arg1), evaluate(arg2))
end

# function evaluate(::typeof(*), arg1::SymbolicValue, arg2::Real)
#     evaluate(*, arg2, arg1)
# end

# function evaluate(::typeof(*), arg1::Real, arg2::SymbolicValue)
#     if arg1 == 1
#         return arg2
#     else
#         BinaryOperation{*}(arg1, arg2)
#     end
# end

function evaluate(::typeof(*), arg1::Zero, arg2::Sym)
    return evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1, arg2::Zero)
    contracting_index = eliminated_indices([get_free_indices(arg1); get_free_indices(arg2)])

    if isempty(contracting_index) # One arg is a scalar or the indices are incompatible.
        return Zero(arg1.indices..., arg2.indices...)
    end

    new_indices = eliminate_indices([get_free_indices(arg1); get_free_indices(arg2)])

    return Zero(new_indices...)
end

function evaluate(::typeof(*), arg1::Zero, arg2::Zero)
    arg1
end

function evaluate(::typeof(+), arg1::Zero, arg2::Zero)
    @assert all(typeof.(arg1.indices) .== typeof.(arg2.indices))

    arg1
end

# TODO: Check that the indices are the same
function evaluate(::typeof(+), arg1::Zero, arg2)
    evaluate(arg2)
end

function evaluate(::typeof(+), arg1, arg2::Zero)
    evaluate(arg1)
end

function evaluate(::typeof(+), arg1, arg2)
    BinaryOperation{+}(arg1, arg2)
end

function evaluate(op::BinaryOperation{*})
    evaluate(*, evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(op::BinaryOperation{+})
    evaluate(+, evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(sym::Union{Sym, KrD, Zero, Real})
    sym
end
