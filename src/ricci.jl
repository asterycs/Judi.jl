import Base.==
import Base.hash
import Base.*
import Base.+
import Base.adjoint

export Upper, Lower
export Sym, KrD, Zero

export equivalent

export flip

export to_string
export to_std_string

Letter = Int

struct Upper
    letter::Letter
end

struct Lower
    letter::Letter
end

LowerOrUpperIndex = Union{Lower, Upper}

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

global const NEXT_LETTER = Ref{Letter}(1)

function get_next_letter()
    tmp = NEXT_LETTER[]
    NEXT_LETTER[] += 1

    return tmp
end

abstract type SymbolicValue end

IndexSet = Vector{LowerOrUpperIndex}

struct Sym <: SymbolicValue
    id::String
    indices::IndexSet

    function Sym(id, indices::LowerOrUpperIndex...)
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

function equivalent(left::Sym, right::Sym)
    return left.id == right.id && all(typeof.(left.indices) .== typeof.(right.indices))
end

struct KrD <: SymbolicValue
    indices::IndexSet

    function KrD(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]
        if isempty(eliminate_indices(indices))
            throw(DomainError(indices, "Indices $indices of δ are invalid"))
        end

        new(indices)
    end
end

function ==(left::KrD, right::KrD)
    return left.indices == right.indices
end

struct Zero <: SymbolicValue
    indices::IndexSet

    function Zero(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        new(indices)
    end
end

function ==(left::Zero, right::Zero)
    return left.indices == right.indices
end

struct BinaryOperation{Op} <: SymbolicValue where Op
    arg1::SymbolicValue
    arg2::SymbolicValue
end

function ==(left::BinaryOperation{Op}, right::BinaryOperation{Op}) where Op
    same_args = left.arg1 == right.arg1 && left.arg2 == right.arg2
    same_args = same_args || (left.arg1 == right.arg2 && left.arg2 == right.arg1)
    return same_args
end

struct UnaryOperation <: SymbolicValue
    op::SymbolicValue
    arg::SymbolicValue
end

function ==(left::UnaryOperation, right::UnaryOperation)
    return left.arg == right.arg && left.op == right.op
end

function _eliminate_indices(arg::IndexSet)
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

    filtered = LowerOrUpperIndex[]

    for e ∈ available
        if !isnothing(e)
            push!(filtered, e)
        end
    end

    return filtered, eliminated
end

function eliminate_indices(arg::IndexSet)
    return first(_eliminate_indices(arg))
end

function eliminated_indices(arg::IndexSet)
    return last(_eliminate_indices(arg))
end

function get_free_indices(arg::Union{Sym, KrD, Zero})
    arg.indices
end

function get_free_indices(arg::UnaryOperation)
    eliminate_indices([get_free_indices(arg.arg); get_free_indices(arg.op)])
end

function get_free_indices(arg::BinaryOperation{*})
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

    pairs = Dict{Letter, Int}()

    for j ∈ arg2_indices
        pairs_in_sweep = Dict{Letter, Int}()

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
    pairs = Dict{Letter, Int}()

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
    pairs = Dict{Letter, Int}()

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
function is_valid_matrix_multiplication(arg1, arg2)
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

    if flip(arg1_indices[end]) != arg2_indices[1]
        return false
    end

    return true
end

function *(arg1::SymbolicValue, arg2::SymbolicValue)
    if isempty(get_free_indices(arg1)) || isempty(get_free_indices(arg2))
        return BinaryOperation{*}(arg1, arg2)
    else
        if !is_contraction_unambigous(arg1, arg2)
            throw(DomainError((arg1, arg2), "Invalid multiplication"))
        end

        arg1_free_indices = get_free_indices(arg1)
        arg2_free_indices = get_free_indices(arg2)

        return BinaryOperation{*}(update_index(arg1, arg1_free_indices[end], flip(arg2_free_indices[1])), arg2)
    end
end

function +(arg1::SymbolicValue, arg2::SymbolicValue)
    BinaryOperation{+}(arg1, arg2)
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

    return BinaryOperation{*}(arg, KrD(flip(from), to))
end

function adjoint(arg::UnaryOperation)
    free_indices = get_free_indices(arg)

    if length(free_indices) >= 1
        throw(DomainError("Adjoint is ambigous"))
    end

    return BinaryOperation{*}(arg, KrD(flip(free_indices[1]), flip(free_indices[1])))
end

function adjoint(arg::BinaryOperation{*})
    free_indices = get_free_indices(arg)

    if length(free_indices) > 1
        throw(DomainError("Adjoint is ambigous"))
    end

    return BinaryOperation{*}(arg, KrD(flip(free_indices[1]), flip(free_indices[1])))
end

function adjoint(arg::BinaryOperation{+})
    return BinaryOperation{+}(adjoint(arg.arg1), adjoint(arg.arg2))
end

function adjoint(arg::Union{Sym, KrD, Zero})
    ids = get_free_indices(arg)

    if length(ids) == 1
        BinaryOperation{*}(arg, KrD(flip(ids[1]), flip(ids[1])))
    elseif length(ids) == 2
        BinaryOperation{*}(BinaryOperation{*}(arg, KrD(flip(ids[2]), flip(ids[2]))), KrD(flip(ids[1]), flip(ids[1])))
    else
        throw(DomainError("Ambgious transpose"))
    end
end

function script(index::Lower)
    @assert index.letter >= 0
    text = []

    for d ∈ reverse(digits(index.letter))
        push!(text, Char(0x2080 + d))
    end

    return join(text)
end

function script(index::Upper)
    @assert index.letter >= 0
    text = []

    for d ∈ reverse(digits(index.letter))
        if d == 0 push!(text, Char(0x2070)) end
        if d == 1 push!(text, Char(0x00B9)) end
        if d == 2 push!(text, Char(0x00B2)) end
        if d == 3 push!(text, Char(0x00B3)) end
        if d > 3 push!(text, Char(0x2070 + d)) end
    end

    return join(text)
end

function to_string(arg::Sym)
    scripts = [script(i) for i ∈ arg.indices]

    arg.id * join(scripts)
end

function to_string(arg::KrD)
    scripts = [script(i) for i ∈ arg.indices]

    "δ" * join(scripts)
end

function to_string(arg::Real)
    string(arg)
end

function to_string(arg::Zero)
    scripts = [script(i) for i ∈ arg.indices]

    "0" * join(scripts)
end

function to_string(arg::UnaryOperation)
    "(" * to_string(arg.arg) * " " * to_string(arg.op) * ")"
end

function to_string(arg::BinaryOperation{Op}) where Op
    "(" * to_string(arg.arg1) * " " * string(Op) * " " * to_string(arg.arg2) * ")"
end

function to_std_string(arg::Sym)
    superscript = ""
    # if length(arg.indices) == 2
    #     if all(i -> typeof(i) == Lower, arg.indices) || all(i -> typeof(i) == Upper, arg.indices)
    #         superscript = "ᵀ"
    #     end
    # else
    if length(arg.indices) == 1
        if all(i -> typeof(i) == Lower, arg.indices)
            superscript = "ᵀ"
        end
    # else
        # @assert false "Tensor format string not implemented"
    end

    return arg.id * superscript
end

function to_std_string(arg::UnaryOperation)
    if typeof(arg.op) == KrD
        if typeof(arg.op) == typeof(arg.op) # is a transpose
            free_indices = get_free_indices(arg.arg)

            if length(free_indices) != 1
                throw(DomainError("Ambigous transpose"))
            end

            @assert flip(free_indices[1]) == arg.op.indices[1]

            return "(" * to_std_string(arg.arg) * ")ᵀ"
        else
            throw(DomainError("Cannot convert expression to std notation"))
        end
    else
        return to_std_string(arg.arg) * to_std_string(arg.op)
    end
end

function to_std_string(arg::BinaryOperation{+})
    return "(" * to_std_string(arg.arg1) * " " * string(+) * " " * to_std_string(arg.arg2) * ")"
end

function to_std_string(arg::BinaryOperation{*})
    free_ids = get_free_indices(arg)

    if !can_contract(arg.arg1, arg.arg2)
        return arg
    elseif length(free_ids) == 1 # result is a vector
        if length(get_free_indices(arg.arg1)) == 2 && length(get_free_indices(arg.arg2)) == 1

            if all(i -> typeof(i) == Lower, get_free_indices(arg.arg1))
                return to_std_string(arg.arg2) * "ᵀ * " * to_std_string(arg.arg1) * "ᵀ"
            end

            if typeof(get_free_indices(arg.arg2)[1]) == Lower
                return to_std_string(arg.arg1) * "ᵀ * " * to_std_string(arg.arg2)
            end

            @assert false "Unreachable"
        elseif length(get_free_indices(arg.arg1)) == 1 && length(get_free_indices(arg.arg2)) == 2
            if all(i -> typeof(i) == Lower, get_free_indices(arg.arg2))
                return "(" * to_std_string(arg.arg1) * " * " * to_std_string(arg.arg2) * ")ᵀ"
            end

            if typeof(get_free_indices(arg.arg1)[1]) == Lower
                return to_std_string(arg.arg1) * " * " * to_std_string(arg.arg2)
            end

            @assert false "Unreachable"
        else
            @assert false "Not implemented"
        end
    end
end