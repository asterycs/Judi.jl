import Base.==
import Base.hash
import Base.*
import Base.+
import Base.adjoint

export Upper, Lower
export Sym, KrD, Zero

export flip

export to_string
export to_std_string

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

function can_contract(arg1::KrD, arg2)
    return can_contract_weak(arg2, arg1)
end

function can_contract(arg1, arg2::KrD)
    return can_contract_weak(arg1, arg2)
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
# TODO: Rename - include "multiplication" or "contraction" in the name.
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

    if flip(arg1.indices[end]) != arg2.indices[1]
        return false
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

    if from == to
        return arg
    end

    @assert from ∈ indices

    if typeof(from) == typeof(flip(to))
        throw(DomainError((from, to), "requested a transpose which isn't allowed"))
    end

    return UnaryOperation(KrD([flip(from); to]), arg)
end
    
function adjoint(arg::SymbolicValue)
    ids = get_free_indices(arg)
    @assert length(ids) == 1
    UnaryOperation(KrD([flip(ids[1]); flip(ids[1])]), arg)
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