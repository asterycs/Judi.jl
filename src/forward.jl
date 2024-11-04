export evaluate

export diff

export to_string
export to_std_string

function diff(sym::Sym, wrt::Sym)
    if sym == wrt
        @assert length(sym.indices) <= 1 # Only scalars and vectors supported for now
        return KrD([sym.indices; lowernext(sym.indices[end])])
    else
        return Zero()
    end
end

function diff(arg::UnaryOperation, wrt::Sym)
    UnaryOperation(arg.op, diff(arg.arg, wrt))
end

function diff(arg::BinaryOperation, wrt::Sym)
    diff(arg.op, arg.arg1, arg.arg2, wrt)
end

function diff(*, arg1, arg2, wrt::Sym)
    BinaryOperation(*, arg1, diff(arg2, wrt)) + BinaryOperation(*, diff(arg1, wrt), arg2)
end

function rotate(op::BinaryOperation)
    println("rotate(op::BinaryOperation)")
    if typeof(op.op) != typeof(*)
        return BinaryOperation(op.op, rotate(op.arg1), rotate(op.arg2))
    end

    return rotate(op.arg1, op.arg2)
end

function rotate(op::UnaryOperation)
    println("rotate(op::UnaryOperation)")
    @show op.op
    @show op.arg
    return UnaryOperation(op.op, rotate(op.arg))
end

function rotate(sym::Sym)
    println("rotate(sym::Sym)")
    return sym
end

function rotate(arg1::Sym, arg2::KrD)
    rotate(arg2, arg1)
end

function rotate(arg1::KrD, arg2::Sym)
    UnaryOperation(arg1, arg2)
end

function rotate(arg1::Sym, arg2::Sym)
    BinaryOperation(*, arg1, arg2)
end

function rotate(arg1::UnaryOperation, arg2::Zero)
    println("rotate(arg1::UnaryOperation, arg2::Zero)")
    return BinaryOperation(*, rotate(arg1), arg2)
end

function rotate(arg1::UnaryOperation, arg2::Sym)
    println("rotate(arg1::UnaryOperation, arg2::Sym)")
    if typeof(arg1.op) == KrD
        return UnaryOperation(arg1.op, rotate(arg1.arg, arg2))
    else
        return BinaryOperation(*, rotate(arg1), arg2)
    end
end

function rotate(arg1::KrD, arg2::UnaryOperation)
    println("rotate(arg1::KrD, arg2::UnaryOperation)")
    rotate(arg2, arg1)
end

function rotate(arg1::UnaryOperation, arg2::KrD)
    println("rotate(arg1::UnaryOperation, arg2::KrD)")
    if typeof(arg1.op) == KrD
        new_krd = evaluate(*, arg1.op, arg2)
        return UnaryOperation(new_krd, rotate(arg1.arg))
    else
        return UnaryOperation(arg1, arg2)
    end
end

function rotate(arg1::KrD, arg2::BinaryOperation)
    println("rotate(arg1::KrD, arg2::BinaryOperation)")
    rotate(arg2, arg1)
end

function rotate(arg1::BinaryOperation, arg2::KrD)
    println("rotate(arg1::BinaryOperation, arg2::KrD)")
    if typeof(arg1.arg1) == KrD
        new_krd = evaluate(*, arg1.arg1, arg2)
        return UnaryOperation(arg1.arg2, new_krd)
    elseif typeof(arg1.arg2) == KrD
        new_krd = evaluate(*, arg1.arg2, arg2)
        return UnaryOperation(arg1.arg1, new_krd)
    else
        return BinaryOperation(*, arg1, arg2)
    end
end

function rotate(arg1::Sym, arg2::BinaryOperation)
    println("rotate(arg1::Sym, arg2::BinaryOperation)")
    rotate(arg2, arg1)
end

function rotate(arg1::BinaryOperation, arg2::Sym)
    println("rotate(arg1::BinaryOperation, arg2::KrD)")
    if typeof(arg1.arg1) == KrD
        return UnaryOperation(arg1.arg1, BinaryOperation(*, arg1.arg2, arg2))
    elseif typeof(arg1.arg2) == KrD
        return UnaryOperation(arg1.arg2, BinaryOperation(*, arg1.arg1, arg2))
    else
        return BinaryOperation(*, rotate(arg1), arg2)
    end
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
        new_arg2 = evaluate(*, arg1.arg2, arg2)
        return BinaryOperation(*, arg1.arg1, new_arg2)
    else
        return UnaryOperation(arg2, arg1)
    end
end

function evaluate(::typeof(*), arg1::Union{Sym, KrD}, arg2::KrD)
    contracting_index = eliminated_indices([get_free_indices(arg1); get_free_indices(arg2)])

    if isempty(contracting_index) # One arg is a scalar
        return UnaryOperation(arg2, arg1)
    end

    @assert length(contracting_index) == 2

    contracting_letter = contracting_index[1].letter

    @assert can_contract(arg1, arg2)

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
    @assert can_contract(arg1, arg2)
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

function to_string(arg::Sym)
    upper_indices = [i.letter for i ∈ arg.indices if typeof(i) == Upper]
    upper_tag = ""
    if !isempty(upper_indices)
        upper_tag = "^(" * string(upper_indices...) * ")"
    end

    lower_indices = [i.letter for i ∈ arg.indices if typeof(i) == Lower]
    lower_tag = ""
    if !isempty(lower_indices)
        lower_tag = "_(" * string(lower_indices...) * ")"
    end

    arg.id * upper_tag * lower_tag
end

function to_string(arg::KrD)
    upper_indices = [i.letter for i ∈ arg.indices if typeof(i) == Upper]
    upper_indices = string(upper_indices...)
    lower_indices = [i.letter for i ∈ arg.indices if typeof(i) == Lower]
    lower_indices = string(lower_indices...)

    "δ" * "^(" * upper_indices * ")_(" * lower_indices * ")"
end

function to_string(arg::Real)
    string(arg)
end

function to_string(arg::Zero)
    "0"
end

function to_string(arg::UnaryOperation)
    "(" * to_string(arg.arg) * " " * to_string(arg.op) * ")"
end

function to_string(arg::BinaryOperation)
    "(" * to_string(arg.arg1) * " " * string(arg.op) * " " * to_string(arg.arg2) * ")"
end

function to_std_string(arg::Sym)
    superscript = ""
    if length(arg.indices) <= 2
        if all(i -> typeof(i) == Lower, arg.indices)
            superscript = "ᵀ"
        end
    else
        @assert false "Tensor format string not implemented"
    end

    return arg.id * superscript
end

function to_std_string(arg::KrD)
    return ""
end

function to_std_string(arg::UnaryOperation)
    return to_std_string(arg.arg) * to_std_string(arg.op)
end

function to_std_string(arg::BinaryOperation)
    separator = string(arg.op)
    term = to_std_string(arg.arg1) * " " * separator * " " * to_std_string(arg.arg2)

    if typeof(arg.op) == typeof(+)
        term = "(" * term * ")"
    elseif typeof(arg.op) == typeof(*)
        # no-op
    else
        @assert false "Not implemented"
    end

    return term
end