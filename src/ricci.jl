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

struct Upper end
struct Lower end

Letter = Int

LowerOrUpperIndex = Union{Lower, Upper}

function flip(_::Lower)
    return Upper()
end

function flip(_::Upper)
    return Lower()
end

function same(_::Lower)
    return Lower()
end

function same(_::Upper)
    return Upper()
end

function ==(left::Lower, right::Lower)
    return true
end

function ==(left::Upper, right::Upper)
    return true
end

function ==(left::Lower, right::Upper)
    return false
end

function ==(left::Upper, right::Lower)
    return false
end

function hash(arg::LowerOrUpperIndex)
    h::UInt = 0

    if typeof(arg) == Lower
        h |= 1
    end

    h
end

abstract type Expression end

IndexSet = Vector{LowerOrUpperIndex}

struct Sym <: Expression
    id::String
    indices::IndexSet

    function Sym(id, indices::LowerOrUpperIndex...)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]

        new(id, indices)
    end
end

function ==(left::Sym, right::Sym)
    return left.id == right.id && left.indices == right.indices
end

struct KrD <: Expression
    indices::IndexSet

    function KrD(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(indices) < 2
            throw(DomainError(indices, "Indices $indices of δ are invalid"))
        end

        new(indices)
    end
end

function ==(left::KrD, right::KrD)
    return left.indices == right.indices
end

struct Zero <: Expression
    indices::IndexSet

    function Zero(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        new(indices)
    end
end

function ==(left::Zero, right::Zero)
    return left.indices == right.indices
end

ContractingPair = Tuple{Int, Int}
Contractions = Vector{ContractingPair}

# TODO: Rename to contraction
struct BinaryOperation{Op} <: Expression where Op
    arg1::Expression
    arg2::Expression
    indices::Contractions
end

# TODO: Check BinaryOperation validity in constructor

function ==(left::BinaryOperation{Op}, right::BinaryOperation{Op}) where Op
    same_args = left.arg1 == right.arg1 && left.arg2 == right.arg2
    same_args = same_args || (left.arg1 == right.arg2 && left.arg2 == right.arg1)
    return same_args
end

struct UnaryOperation <: Expression
    op::Expression
    arg::Expression
end

function ==(left::UnaryOperation, right::UnaryOperation)
    return left.arg == right.arg && left.op == right.op
end

function eliminate_indices(arg1::IndexSet, arg2::IndexSet, indices)
    arg1 = Union{LowerOrUpperIndex, Nothing}[i for i ∈ arg1]
    arg2 = Union{LowerOrUpperIndex, Nothing}[i for i ∈ arg2]

    for pair ∈ indices
        if typeof(flip(arg1[pair[1]])) == typeof(arg2[pair[2]])
            arg1[pair[1]] = nothing
            arg2[pair[2]] = nothing
        end
    end

    remaining = LowerOrUpperIndex[]

    for i ∈ arg1
        if !isnothing(i)
            push!(remaining, i)
        end
    end

    for i ∈ arg2
        if !isnothing(i)
            push!(remaining, i)
        end
    end

    @assert length(remaining) < length(arg1) + length(arg2)

    return remaining
end

function eliminated_indices(arg1::IndexSet, arg2::IndexSet, indices)
    arg1 = Union{LowerOrUpperIndex, Nothing}[i for i ∈ arg1]
    arg2 = Union{LowerOrUpperIndex, Nothing}[i for i ∈ arg2]

    eliminated = Contractions()

    for pair ∈ indices
        if typeof(flip(arg1[pair[1]])) == typeof(arg2[pair[2]])
            push!(eliminated, pair)
        end
    end

    return eliminated
end

function get_free_indices(arg::Union{Sym, KrD, Zero})
    arg.indices
end

function get_free_indices(arg::UnaryOperation)
    @assert false "Not implemented"
end

function get_free_indices(arg::BinaryOperation{*})
    eliminate_indices(get_free_indices(arg.arg1), get_free_indices(arg.arg2), arg.indices)
end

function can_contract(arg1, arg2, indices::Contractions)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    for pair ∈ indices
        if typeof(arg1_indices[pair[1]]) == typeof(arg2_indices[pair[2]])
            return false
        end
    end

    return true
end

function is_valid_multiplication(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    if length(arg1_indices) <= 2 && length(arg2_indices) <= 2
        if typeof(arg1_indices[end]) == Lower && typeof(arg2_indices[1]) == Upper
            return true
        end
    end

    return false
end

function *(arg1, arg2)
    arg1_free_indices = get_free_indices(arg1)
    arg2_free_indices = get_free_indices(arg2)

    if isempty(arg1_free_indices) || isempty(arg2_free_indices)
        @assert false "Not implemented"
        return BinaryOperation{*}(arg1, arg2)
    else
        if !is_valid_multiplication(arg1, arg2)
            throw(DomainError((arg1, arg2), "Invalid multiplication"))
        end

        return BinaryOperation{*}(arg1, arg2, [(length(arg1_free_indices), 1)])
    end
end

function +(arg1::Expression, arg2::Expression)
    BinaryOperation{+}(arg1, arg2)
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
        BinaryOperation{*}(arg, KrD(flip(ids[1]), flip(ids[1])), [(1, 1)])
    elseif length(ids) == 2
        BinaryOperation{*}(BinaryOperation{*}(arg, KrD(flip(ids[2]), flip(ids[2])), [(2, 1)]), KrD(flip(ids[1]), flip(ids[1])), [(1, 1)])
    else
        throw(DomainError("Ambgious transpose"))
    end
end

function script(_::Lower, letter::Letter)
    text = []

    for d ∈ reverse(digits(letter))
        push!(text, Char(0x2080 + d))
    end

    return join(text)
end

function script(_::Upper, letter::Letter)
    text = []

    for d ∈ reverse(digits(letter))
        if d == 0 push!(text, Char(0x2070)) end
        if d == 1 push!(text, Char(0x00B9)) end
        if d == 2 push!(text, Char(0x00B2)) end
        if d == 3 push!(text, Char(0x00B3)) end
        if d > 3 push!(text, Char(0x2070 + d)) end
    end

    return join(text)
end

function to_string(arg::Sym)
    arg.id
end

function to_string(arg::KrD)
    "δ"
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

function to_string(arg::BinaryOperation{*})
    to_string(arg.arg1, arg.arg2, arg.indices)
end

function to_string(arg1, arg2, contractions::Contractions)
    arg1_contracting_indices = [p[1] for p ∈ contractions]
    arg2_contracting_indices = [p[2] for p ∈ contractions]

    next_free_index = length(contractions) + 1
    next_dummy_index = 1

    out = to_string(arg1)
    if typeof(arg1) != Sym && typeof(arg1) != KrD
        out = "(" * out * ")"
    end

    for (i,ul) ∈ enumerate(get_free_indices(arg1))
        if i ∈ arg1_contracting_indices
            out *= script(ul, next_dummy_index)
            next_dummy_index += 1
        else
            out *= script(ul, next_free_index)
            next_free_index += 1
        end
    end

    out *= to_string(arg2)
    next_dummy_index = 1

    for (i,ul) ∈ enumerate(arg2.indices)
        if i ∈ arg2_contracting_indices
            out *= script(ul, next_dummy_index)
            next_dummy_index += 1
        else
            out *= script(ul, next_free_index)
            next_free_index += 1
        end
    end

    return out
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