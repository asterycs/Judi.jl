export evaluate

export D

function D(expr, wrt::Sym)
    linear_form = diff(expr, wrt)
    linear_form = evaluate(linear_form)

    linear_form
end

function diff(sym::Sym, wrt::Sym)
    if sym == wrt
        @assert length(sym.indices) <= 1 # Only scalars and vectors supported for now
        return KrD(sym.indices..., lowernext(sym.indices[end]))
    else
        return Zero(sym.indices..., lowernext(sym.indices[end]))
    end
end

function diff(arg::UnaryOperation, wrt::Sym)
    UnaryOperation(arg.op, diff(arg.arg, wrt))
end

function diff(arg::BinaryOperation, wrt::Sym)
    diff(arg.op, arg.arg1, arg.arg2, wrt)
end

function diff(::typeof(*), arg1, arg2, wrt::Sym)
    BinaryOperation(+, BinaryOperation(*, arg1, diff(arg2, wrt)), BinaryOperation(*, diff(arg1, wrt), arg2))
end

function evaluate(sym::Sym)
    sym
end

function evaluate(::typeof(*), arg1::BinaryOperation, arg2::BinaryOperation)
    evaluate(*, evaluate(arg1), evaluate(arg2))
end

function evaluate(arg::UnaryOperation)
    if typeof(arg.op) == KrD
        return evaluate(*, evaluate(arg.arg), arg.op)
    end

    return arg
end

function evaluate(::typeof(*), arg1::KrD, arg2::Sym)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::KrD, arg2::BinaryOperation)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::BinaryOperation, arg2::KrD)
    if typeof(arg1.op) == typeof(*)
        if can_contract(arg1.arg2, arg2)
            new_arg2 = evaluate(*, arg1.arg2, arg2)
            return BinaryOperation(*, arg1.arg1, new_arg2)
        elseif can_contract(arg1.arg1, arg2)
            new_arg1 = evaluate(*, arg1.arg1, arg2)
            return BinaryOperation(*, new_arg1, arg1.arg2)
        else
            return BinaryOperation(*, arg1, arg2)
        end
    else
        return UnaryOperation(arg2, arg1)
    end
end

function evaluate(::typeof(*), arg1::Union{Sym, KrD}, arg2::KrD)
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
    return BinaryOperation(*, arg1, arg2)
end

function evaluate(::typeof(*), arg1::SymbolicValue, arg2::Real)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::Real, arg2::SymbolicValue)
    if arg1 == 1
        return arg2
    else
        BinaryOperation(*, arg1, arg2)
    end
end

function evaluate(::typeof(*), arg1, arg2::Zero)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::Zero, arg2)
    arg1
end

function evaluate(::typeof(+), arg1::Zero, arg2)
    arg2
end

function evaluate(::typeof(+), arg1, arg2::Zero)
    arg1
end

function evaluate(::typeof(+), arg1, arg2)
    BinaryOperation(+, arg1, arg2)
end

function evaluate(op::BinaryOperation)
    evaluate(op.op, evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(sym::Union{Sym, KrD, Zero, Real})
    sym
end
