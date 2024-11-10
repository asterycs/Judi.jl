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
        return KrD(sym.indices..., Lower())
    else
        return Zero(sym.indices..., Lower())
    end
end

function diff(arg::KrD, wrt::Sym)
    return Zero(arg.indices..., Lower())
end

function diff(arg::UnaryOperation, wrt::Sym)
    UnaryOperation(arg.op, diff(arg.arg, wrt))
end

function diff(arg::Contraction, wrt::Sym)
    Sum(Contraction(arg.arg1, diff(arg.arg2, wrt), arg.indices), Contraction(diff(arg.arg1, wrt), arg.arg2, arg.indices))
end

function diff(arg::Sum, wrt::Sym)
    Sum(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function mirror(contractions::Contractions)
    return [(pair[2], pair[1]) for pair ∈ contractions]
end

function evaluate(sym::Sym)
    sym
end

function evaluate(arg::UnaryOperation)
    @assert false "Not implemented"
end

function evaluate(::typeof(*), arg1::KrD, arg2::Sym, contractions::Contractions)
    evaluate(*, arg2, arg1, mirror(contractions))
end

function evaluate(::typeof(*), arg1::Union{Sym, KrD}, arg2::KrD, contractions::Contractions)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    if !can_contract(arg1, arg2, contractions)
        return Contraction(arg1, arg2, contractions)
    end

    @assert length(contractions) == 1 "Trace not implemented"
    @assert length(arg2.indices) == 2 "Generalized δ not implemented"

    new_indices = LowerOrUpperIndex[]

    for (i,idx) ∈ enumerate(arg1_indices)
        for (from,to) ∈ contractions
            if i == from
                new_index = to == 1 ? 2 : 1
                push!(new_indices, arg2_indices[new_index])
            else
                push!(new_indices, idx)
            end
        end
    end


    if typeof(arg1) == Sym
        return Sym(arg1.id, new_indices...)
    elseif typeof(arg1) == KrD
        return KrD(new_indices...)
    else
        @assert false "Unreachable"
    end
end

function evaluate(::typeof(*), arg1, arg2, contractions::Contractions)
    return Contraction(evaluate(arg1), evaluate(arg2), contractions)
end

function evaluate(::typeof(*), arg1::Expression, arg2::Real)
    @assert false "Not implemented"
end

function evaluate(::typeof(*), arg1::Real, arg2::Expression)
    @assert false "Not implemented"
end

function evaluate(::typeof(*), arg1, arg2::Zero, contractions::Contractions)
    evaluate(*, arg2, arg1, mirror(contractions))
end

function evaluate(::typeof(*), arg1::Zero, arg2, contractions::Contractions)
    arg1
end

function evaluate(::typeof(+), arg1::Zero, arg2::Zero)
    @assert all(typeof.(arg1.indices) .== typeof.(arg2.indices))

    arg1
end

function evaluate(::typeof(+), arg1::Zero, arg2)
    arg2
end

function evaluate(::typeof(+), arg1, arg2::Zero)
    arg1
end

function evaluate(::typeof(+), arg1, arg2)
    Sum(arg1, arg2)
end

function evaluate(op::Contraction)
    evaluate(*, evaluate(op.arg1), evaluate(op.arg2), op.indices)
end

function evaluate(op::Sum)
    evaluate(+, evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(sym::Union{Sym, KrD, Zero, Real})
    sym
end
