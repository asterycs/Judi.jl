using Revise

import Base.==
import Base.hash
import Base.*
import Base.+
import Base.adjoint

export Upper, Lower
export Sym, KrD, Zero

export flip
export evaluate

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

function lowernext(index::Upper)
    return Lower(index.letter + 1)
end

function lowernext(index::Lower)
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

function ==(left::LowerOrUpperIndex, right::LowerOrUpperIndex)
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

    function Sym(id, indices)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]
        if !isempty(eliminated_indices(indices))
            throw(DomainError(indices, "Indices of $id are invalid"))
        end

        letters = unique([i.letter for i ∈ indices])
        if length(letters) != length(indices)
            throw(DomainError(indices, "Indices of $id are invalid"))
        end

        new(id, indices)
    end
end

function ==(left::Sym, right::Sym)
    return left.id == right.id && left.indices == right.indices
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

function ==(left::BinaryOperation, right::BinaryOperation)
    same_args = left.arg1 == right.arg1 && left.arg2 == right.arg2
    same_args = same_args || (left.arg1 == right.arg2 && left.arg2 == right.arg1)
    return left.op == right.op && same_args
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
    available = Union{Nothing, Lower, Upper}[i for i ∈ arg]
    eliminated = LowerOrUpperIndex[]

    for i ∈ eachindex(available)
        if isnothing(available[i])
            continue
        end

        for j ∈ eachindex(available)
            if isnothing(available[j])
                continue
            end

            if flip(available[j]) == available[i]
                push!(eliminated, available[i])
                push!(eliminated, available[j])

                available[i] = nothing
                available[j] = nothing
            end
        end
    end

    return eliminated
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

function are_indices_unique(indices::IndexSet)
    return length(unique(indices)) == length(indices)
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

function can_contract(arg1::KrD, arg2::Sym)
    return can_contract_weak(arg2, arg1)
end

function can_contract(arg1::Sym, arg2::KrD)
    return can_contract_weak(arg1, arg2)
end

function can_contract(arg1::KrD, arg2::KrD)
    return can_contract_weak(arg1, arg2) || can_contract_weak(arg2, arg1)
end

function can_contract_weak(arg1, arg2::KrD)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # If there is a matching index pair then the contraction is unambigous.
    # Duplicate pairs such that several indices in arg2 matches one index in
    # arg1 are allowed.

    pairs = Dict{Letter, Int64}()

    for j ∈ arg2_indices
        pairs_in_sweep = Dict{Letter, Int64}()

        for i ∈ arg1_indices
            if flip(i) == j
                if haskey(pairs_in_sweep, i.letter)
                    return false
                else
                    pairs_in_sweep[i.letter] = 1

                    if haskey(pairs, i.letter)
                        pairs[i.letter] += 1
                    else
                        pairs[i.letter] = 0
                    end
                end
            end
        end
    end

    if length(pairs) == 1
        return true
    end

    return false
end

function can_contract(arg1::Sym, arg2::Sym)
    return can_contract_strong(arg2, arg1)
end

function can_contract(arg1, arg2::Sym)
    return can_contract_strong(arg2, arg1)
end

function can_contract(arg1::Sym, arg2)
    return can_contract_strong(arg1, arg2)
end

function can_contract_strong(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # If there is exactly one matching index pair then the contraction is unambigous.
    pairs = Dict{Letter, Int64}()

    for i ∈ arg1_indices
        for j ∈ arg2_indices
            if flip(i) == j
                if haskey(pairs, i.letter)
                    pairs[i.letter] += 1
                else
                    pairs[i.letter] = 1
                end
            end
        end
    end

    if length(pairs) == 1 && first(values(pairs)) == 1
        return true
    end

    return false
end

function is_contraction_unambigous(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # Check that indices are unique.
    if !are_indices_unique(arg1_indices) || !are_indices_unique(arg2_indices)
        return false
    end

    # Check that there are no contractions within the arguments.
    if !isempty(eliminated_indices(arg1_indices)) || !isempty(eliminated_indices(arg2_indices))
        return false
    end

    # Otherwise, if there is exactly one matching index pair then the contraction is unambigous.
    pairs = Dict{Letter, Int64}()

    for i ∈ arg1_indices
        for j ∈ arg2_indices
            if flip(i) == j
                if haskey(pairs, i.letter)
                    pairs[i.letter] += 1
                else
                    pairs[i.letter] = 1
                end
            end
        end
    end

    if length(pairs) == 1 && first(values(pairs)) == 1
        return true
    end

    # Otherwise, if there is an unambigous Upper/Lower pair then the
    # contraction can be done with updated indices.
    # We don't updated the indices here.
    if length(arg1_indices) == 1 || length(arg2_indices) == 1
        for i ∈ arg1_indices
            for j ∈ arg2_indices
                if typeof(flip(i)) == typeof(j)
                    return true
                end
            end
        end
    end

    return false
end

# If any other except arg1.indices[end] and arg2.indices[1] match
# then the input does not correspond to valid standard notation.
function is_valid_standard_notation(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    if length(arg1_indices) == 1
        arg1_indices = [arg1_indices; arg1_indices]
    end

    if length(arg2_indices) == 1
        arg2_indices = [arg2_indices; arg2_indices]
    end

    for i ∈ arg1_indices[1:end-1]
        for j ∈ arg2_indices[2:end]
            if i.letter == j.letter
                return false
            end
        end
    end

    return true
end

function *(arg1, arg2)
    if isempty(get_free_indices(arg1)) || isempty(get_free_indices(arg2))
        return BinaryOperation(*, arg1, arg2)
    else
        if !is_contraction_unambigous(arg1, arg2)
            throw(DomainError((arg1, arg2), "Invalid multiplication"))
        end

        if !is_valid_standard_notation(arg1, arg2)
            throw(DomainError("Input operands do not correspond to valid standard notation"))
        end

        arg1_free_indices = get_free_indices(arg1)
        arg2_free_indices = get_free_indices(arg2)

        return BinaryOperation(*, update_index(arg1, arg1_free_indices[end], flip(arg2_free_indices[1])), arg2)
    end
end

function +(arg1::SymbolicValue, arg2::SymbolicValue)
    BinaryOperation(+, arg1, arg2)
end

function update_index(arg, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    indices = get_free_indices(arg)

    @assert !isempty(indices)

    if from == to
        return arg
    end

    update_index_impl(arg, from, to)[1]
end

function update_index_impl(arg::UnaryOperation, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    @assert typeof(arg.op) == KrD # only KrD supported for now

    # TODO: Simplify this; verify that the order of the KrD:s can be changed

    if arg.op.indices[1] == from # this UnaryOperation alters the index that should be changed
        if arg.op.indices[1] == arg.op.indices[2] # is a transpose
            from = flip(from)
            to = flip(to)
        end
    end

    new_arg,index_map = update_index_impl(arg.arg, from, to)

    new_op,_ = update_index_impl(arg.op, index_map)

    return (UnaryOperation(new_op, new_arg), index_map)
end

function update_index_impl(arg::Sym, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    if typeof(from) == typeof(flip(to))
        throw(DomainError((from, to), "requested a transpose which isn't allowed"))
    end

    if flip(from) == to
        return arg, Dict{Letter, Letter}()
    end

    return UnaryOperation(KrD([flip(from); to]), arg), Dict{Letter, Letter}(from.letter => to.letter)
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
    BinaryOperation(*, evaluate(arg1), evaluate(arg2))
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

function evaluate(::typeof(*), arg1, arg2::KrD)
    contracting_index = eliminated_indices([arg1.indices; arg2.indices])

    if isempty(contracting_index)
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

# x = Sym("x", [Upper(2)])
# A = Sym("A", [Upper(1); Lower(2)])

# to_string(x' * A * x)

# to_string((diff(x' * A * x, x)))
# to_string(evaluate(diff(x' * A * x)))