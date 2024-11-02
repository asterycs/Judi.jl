export evaluate

export diff

export to_string

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
