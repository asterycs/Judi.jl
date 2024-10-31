using Revise

import Base.==
import Base.hash
import Base.*
import Base.+
import Base.adjoint

export Upper, Lower
export Sym, KrD, Zero

export flip
export eliminate_indices, eliminated_indices

export to_string

Letter = Int64

mutable struct Upper
    letter::Letter
end

mutable struct Lower
    letter::Letter
end

LowerOrUpperIndex = Union{Lower, Upper}

function flipnext(index::Lower)
    return Upper(index.letter + 1)
end

function flipnext(index::Upper)
    return Lower(index.letter + 1)
end

function flip(index::Lower)
    return Upper(index.letter)
end

function flip(index::Upper)
    return Lower(index.letter)
end

function same(old::Lower, letter::Letter)
    return Lower(letter)
end

function same(old::Upper, letter::Letter)
    return Upper(letter)
end

function (==)(left::LowerOrUpperIndex, right::LowerOrUpperIndex)
    if typeof(left) == typeof(right)
        if left.letter == right.letter
            return true
        end
    end

    return false
end

function hash(arg::LowerOrUpperIndex)
    h::UInt = 0

    if typeof(arg) == Lower
        h |= 1
    end

    h |= arg.letter << 1

    h
end

abstract type SymbolicValue end

IndexSet = Vector{LowerOrUpperIndex}

struct Sym <: SymbolicValue
    id::String
    indices::IndexSet
    derivative::SymbolicValue
end

function ==(left::Sym, right::Sym)
    return left.id == right.id && left.indices == right.indices && left.derivative == right.derivative
end

struct KrD <: SymbolicValue
    indices::IndexSet
end

function ==(left::KrD, right::KrD)
    return left.indices == right.indices
end

struct Zero <: SymbolicValue
end

mutable struct BinaryOperation <: SymbolicValue
    op
    arg1::SymbolicValue
    arg2::SymbolicValue
end

mutable struct UnaryOperation <: SymbolicValue
    op::SymbolicValue
    arg::SymbolicValue
end

function ==(left::UnaryOperation, right::UnaryOperation)
    return left.arg == right.arg && left.op == right.op
end

function eliminate_indices(arg::IndexSet)
    available = Union{Nothing, Lower, Upper}[i for i ∈ arg]

    for i ∈ eachindex(available)
        if isnothing(available[i])
            continue
        end

        for j ∈ eachindex(available)
            if isnothing(available[j])
                continue
            end

            if flip(available[j]) == available[i]
                available[i] = nothing
                available[j] = nothing
            end
        end
    end

    filtered = LowerOrUpperIndex[]

    for e ∈ available
        if !isnothing(e)
            push!(filtered, e)
        end
    end

    return filtered
end

function eliminated_indices(arg::IndexSet)
    output = Letter[]

    for i ∈ arg
        has_pair = false
        for j ∈ arg
            if i === j
                continue
            end

            if flip(i) == j
                has_pair = true
            end
        end

        if has_pair
            push!(output, i.letter)
        end
    end

    return unique(output)
end

function get_free_indices(arg::Union{Sym, KrD})
    arg.indices
end

function get_free_indices(arg::UnaryOperation)
    eliminate_indices([get_free_indices(arg.arg); get_free_indices(arg.op)])
end

function get_free_indices(arg::BinaryOperation)
    eliminate_indices([get_free_indices(arg.arg1); get_free_indices(arg.arg2)])
end

function can_contract(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    if isempty(arg1_indices) || isempty(arg2_indices)
        return true
    end


    for i ∈ arg1_indices
        for j ∈ arg2_indices
            if flip(i) == j
                return true
            end
        end
    end

    return false
end

function can_contract(arg1, arg2, index::Letter)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    for i ∈ arg1_indices
        if i.letter == index
            for j ∈ arg2_indices
                if flip(i) == j
                    return true
                end
            end
        end
    end

    return false
end

function *(arg1::SymbolicValue, arg2::SymbolicValue)
    if isempty(get_free_indices(arg1)) || isempty(get_free_indices(arg2))
        return BinaryOperation(*, arg1, arg2)
    else
        return BinaryOperation(*, update_index(arg1, flip(arg2.indices[1])), arg2)
    end
end

function +(arg1::SymbolicValue, arg2::SymbolicValue)
    BinaryOperation(+, arg1, arg2)
end

# TODO: Wrong index might be updated for matrices!!!
function update_index(arg, index::LowerOrUpperIndex)
    indices = get_free_indices(arg)

    @assert !isempty(indices)

    if index == indices[end]
        arg
    else
        update_index_impl(arg, index)[1]
    end
end

function update_index_impl(arg::UnaryOperation, index::LowerOrUpperIndex)
    new_arg,index_map = update_index_impl(arg.arg, index)

    new_op,_ = update_index_impl(arg.op, index_map)

    return (UnaryOperation(new_op, new_arg), index_map)
end

function update_index_impl(arg::Sym, index::LowerOrUpperIndex)
    old_index = arg.indices[end]
    if old_index == index
        return arg, Dict{Letter, Letter}()
    end

    return UnaryOperation(KrD([flip(old_index); index]), arg), Dict{Letter, Letter}(old_index.letter => index.letter)
end

function update_index_impl(arg::KrD, index_map::Dict)
    newarg = deepcopy(arg)

    for i ∈ eachindex(newarg.indices)
        if haskey(index_map, newarg.indices[i].letter)
            newarg.indices[i].letter = index_map[newarg.indices[i].letter]
        end
    end

    (newarg, index_map)
end
    
function adjoint(arg::SymbolicValue)
    ids = get_free_indices(arg)
    @assert length(ids) == 1
    UnaryOperation(KrD([flip(ids[1]); flip(ids[1])]), arg)
end

function diff(sym::Sym)
    sym.derivative
end

function diff(arg::UnaryOperation)
    UnaryOperation(arg.op, diff(arg.arg))
end

function diff(arg::BinaryOperation)
    diff(arg.op, arg.arg1, arg.arg2)
end

function diff(*, arg1, arg2)
    BinaryOperation(*, arg1, diff(arg2)) + BinaryOperation(*, diff(arg1), arg2)
end

function evaluate(sym::Sym)
    sym
end

function evaluate(::typeof(*), arg1::BinaryOperation, arg2::BinaryOperation)
    BinaryOperation(*, evaluate(arg1), evaluate(arg2))
end

function evaluate(arg::UnaryOperation)
    if typeof(arg.op) == KrD
        return evaluate(*, evaluate(arg.arg), arg.op)
    end

    return arg
end

function evaluate(::typeof(*), arg1::Union{Sym, KrD}, arg2::KrD)
    contracting_index = eliminated_indices([arg1.indices; arg2.indices])

    @assert length(contracting_index) == 1

    contracting_index = contracting_index[1]

    @assert can_contract(arg1, arg2, contracting_index)

    @assert length(arg2.indices) == 2

    newarg = deepcopy(arg1)

    for i ∈ eachindex(newarg.indices)
        if newarg.indices[i].letter == arg2.indices[1].letter
            newarg.indices[i] = arg2.indices[2]
        end
    end

    newarg
end

function evaluate(::typeof(*), arg1, arg2)
    @assert can_contract(arg1, arg2)
    return BinaryOperation(*, arg1, arg2)
end

function evaluate(::typeof(*), arg1::Real, arg2::SymbolicValue)
    if arg1 == 1
        return arg2
    else
        BinaryOperation(*, arg1, arg2)
    end
end

function evaluate(::typeof(*), arg1::Sym, arg2::Real)
    evaluate(*, arg2, arg1)
end

function evaluate(::typeof(*), arg1::Zero, arg2)
    arg1
end

function evaluate(::typeof(*), arg1, arg2::Zero)
    evaluate(*, arg2, arg1)
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

# x = Sym("x", [Upper(2)], KrD([Upper(2); Lower(3)]))
# A = Sym("A", [Upper(1); Lower(2)], Zero([]))

# to_string(x' * A)

# to_string((diff(x' * A)))
# to_string(evaluate(diff(x' * A)))