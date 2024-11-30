import Base.==
import Base.hash
import Base.*
import Base.+
import Base.adjoint
import Base.show

export Upper, Lower
export Tensor, KrD, Zero

export tr
export Sin, Cos

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

LowerOrUpperIndex = Union{Lower,Upper}

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

function hash(arg::LowerOrUpperIndex)
    h::UInt = 0

    if typeof(arg) == Lower
        h |= 1
    end

    h |= arg.letter << 1

    h
end

const global NEXT_LETTER = Ref{Letter}(1)

function get_next_letter()
    tmp = NEXT_LETTER[]
    NEXT_LETTER[] += 1

    return tmp
end

abstract type TensorValue end

# Shortcut for simpler comparison from
# https://stackoverflow.com/questions/62336686/struct-equality-with-arrays
function ==(a::T, b::T) where {T<:TensorValue}
    f = fieldnames(T)

    return (getfield.(Ref(a), f) == getfield.(Ref(b), f)) ||
           (reverse(getfield.(Ref(a), f)) == getfield.(Ref(b), f))
end

Value = Union{TensorValue,Number}
IndexSet = Vector{LowerOrUpperIndex}

function get_free_indices(arg::Number)
    return LowerOrUpperIndex[]
end

function equivalent(arg1::Number, arg2::Number)
    return arg1 == arg2
end

struct Tensor <: TensorValue
    id::String
    indices::IndexSet

    function Tensor(id, indices::LowerOrUpperIndex...)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]

        letters = unique([i.letter for i ∈ indices])
        if length(letters) != length(indices)
            throw(DomainError(indices, "Indices of $id are invalid"))
        end

        new(id, indices)
    end
end

function equivalent(left::Tensor, right::Tensor)
    return left.id == right.id && all(typeof.(left.indices) .== typeof.(right.indices))
end

struct KrD <: TensorValue
    indices::IndexSet

    function KrD(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        new(indices)
    end
end

function equivalent(left::KrD, right::KrD)
    return all(typeof.(left.indices) .== typeof.(right.indices))
end

struct Zero <: TensorValue
    indices::IndexSet

    function Zero(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        new(indices)
    end
end

struct BinaryOperation{Op} <: TensorValue where {Op}
    arg1::Value
    arg2::Value
end

function equivalent(left::BinaryOperation, right::BinaryOperation)
    if typeof(left) != typeof(right)
        return false
    end

    same_types =
        (
            typeof(left.arg1) == typeof(right.arg1) &&
            typeof(left.arg2) == typeof(right.arg2)
        ) ||
        (typeof(left.arg1) == typeof(right.arg2) && typeof(left.arg2) == typeof(right.arg1))

    return same_types &&
           (equivalent(left.arg1, right.arg1) && equivalent(left.arg2, right.arg2)) ||
           (equivalent(left.arg1, right.arg2) && equivalent(left.arg2, right.arg1))
end

abstract type UnaryOperation <: TensorValue end

function equivalent(arg1::T, arg2::T) where {T<:UnaryOperation}
    return equivalent(arg1.arg, arg2.arg)
end

struct Sin <: UnaryOperation
    arg::TensorValue
end

struct Cos <: UnaryOperation
    arg::TensorValue
end

NonTrivialValue = Union{Tensor,KrD,BinaryOperation{*},BinaryOperation{+},Real}
# TODO: Rename BinaryOperation{*} and align with Mult below
NonTrivialNonMult = Union{Tensor,KrD,BinaryOperation{+},Real}

function consolidate(indices::Vector{Union{Nothing,Lower,Upper}})
    filtered = LowerOrUpperIndex[]

    for e ∈ indices
        if !isnothing(e)
            push!(filtered, e)
        end
    end

    return filtered
end

function _eliminate_indices(arg1::IndexSet, arg2::IndexSet)
    CanBeNothing = Union{Nothing,Lower,Upper}
    available1 = CanBeNothing[i for i ∈ arg1]
    available2 = CanBeNothing[i for i ∈ arg2]
    eliminated = LowerOrUpperIndex[]

    for i ∈ eachindex(available1)
        if isnothing(available1[i])
            continue
        end

        for j ∈ eachindex(available2)
            if isnothing(available2[j])
                continue
            end

            if flip(available2[j]) == available1[i]
                push!(eliminated, available1[i])
                push!(eliminated, available2[j])
                available1[i] = nothing
                available2[j] = nothing
            end
        end
    end

    filtered1 = consolidate(available1)
    filtered2 = consolidate(available2)

    return (filtered1, filtered2), eliminated
end

function eliminate_indices(arg1::IndexSet, arg2::IndexSet)
    return first(_eliminate_indices(arg1, arg2))
end

function eliminated_indices(arg1::IndexSet, arg2::IndexSet)
    return last(_eliminate_indices(arg1, arg2))
end

function are_indices_equivalent(arg1::IndexSet, arg2::IndexSet)

    if length(arg1) != length(arg2)
        return false
    end

    U = union(arg1, arg2)

    if length(U) == length(arg1)
        return true
    end

    return false
end

function are_indices_equivalent(arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return are_indices_equivalent(arg1_indices, arg2_indices)
end

function get_free_indices(arg::Union{Tensor,KrD,Zero})
    arg.indices
end

function get_free_indices(arg::Cos)
    return get_free_indices(arg.arg)
end

function get_free_indices(arg::BinaryOperation{*})
    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg.arg1), get_free_indices(arg.arg2))

    return [arg1_free_indices; arg2_free_indices]
end

function get_free_indices(arg::BinaryOperation{+})
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert are_indices_equivalent(arg1_ids, arg2_ids)

    return arg1_ids
end

function are_indices_unique(indices::IndexSet)
    return length(unique(indices)) == length(indices)
end

function can_contract(arg1::Real, arg2::TensorValue)
    return false
end

function can_contract(arg1::TensorValue, arg2::Real)
    return false
end

function can_contract(arg1::KrD, arg2::TensorValue)
    return can_contract_weak(arg2, arg1)
end

function can_contract(arg1::TensorValue, arg2::KrD)
    return can_contract_weak(arg1, arg2)
end

function can_contract(arg1::KrD, arg2::Tensor)
    return can_contract_weak(arg2, arg1)
end

function can_contract(arg1::Tensor, arg2::KrD)
    return can_contract_weak(arg1, arg2)
end

function can_contract(arg1::KrD, arg2::KrD)
    return can_contract_weak(arg1, arg2) || can_contract_weak(arg2, arg1)
end

# TODO: Rename to _can_contract(arg1::TensorValue, arg2::KrD)
function can_contract_weak(arg1::TensorValue, arg2::KrD)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # If there is a matching index pair then the contraction is unambigous.
    # Duplicate pairs such that several indices in arg2 matches one index in
    # arg1 are allowed.

    pairs = Dict{Letter,Int}()

    for j ∈ arg2_indices
        pairs_in_sweep = Dict{Letter,Int}()

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

    if length(pairs) == 1 || length(pairs) == 2
        return true
    end

    return false
end

function can_contract(arg1::Tensor, arg2::Tensor)
    return can_contract_strong(arg2, arg1)
end

function can_contract(arg1::TensorValue, arg2::Tensor)
    return can_contract_strong(arg2, arg1)
end

function can_contract(arg1::Tensor, arg2::TensorValue)
    return can_contract_strong(arg1, arg2)
end

function can_contract_strong(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # If there is exactly one matching index pair then the contraction is unambigous.
    pairs = Dict{Letter,Int}()

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

    # Otherwise, if there is exactly one matching index pair then the contraction is unambigous.
    pairs = Dict{Letter,Int}()

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

    # Otherwise, check if this is a matrix-matrix or a matrix-vector multiplication.
    # We don't updated the indices here.
    if length(arg1_indices) <= 2 && length(arg2_indices) <= 2
        if typeof(flip(arg1_indices[end])) == typeof(arg2_indices[1])
            return true
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

function tr(arg::TensorValue)
    free_ids = get_free_indices(arg)

    de = DomainError("Trace is defined only for matrices")

    if length(free_ids) != 2
        throw(de)
    end

    if all(typeof.(free_ids) .== Upper) || all(typeof.(free_ids) .== Lower)
        throw(de)
    end

    return BinaryOperation{*}(arg, KrD(flip(free_ids[2]), flip(free_ids[1])))
end

function *(arg1::TensorValue, arg2::Number)
    return arg2 * arg1
end

function *(arg1::Value, arg2::TensorValue)
    arg1_free_indices = get_free_indices(arg1)
    arg2_free_indices = get_free_indices(arg2)

    if isempty(arg1_free_indices) || isempty(arg2_free_indices)
        return BinaryOperation{*}(arg1, arg2)
    end

    if length(arg1_free_indices) > 2
        throw(DomainError(arg1, "Multiplication involving tensor \"$arg1\" is ambiguous"))
    end

    if length(arg2_free_indices) > 2
        throw(DomainError(arg2, "Multiplication involving tensor \"$arg2\" is ambiguous"))
    end

    if typeof(arg1_free_indices[end]) == Lower && typeof(arg2_free_indices[1]) == Upper
        return BinaryOperation{*}(
            update_index(arg1, arg1_free_indices[end], flip(arg2_free_indices[1])),
            arg2,
        )
    end

    if length(arg1_free_indices) == 1 && length(arg2_free_indices) == 1 &&
        typeof(arg1_free_indices[end]) == Upper && typeof(arg2_free_indices[1]) == Lower
        return BinaryOperation{*}(
            arg1,
            update_index(arg2, arg2_free_indices[end], same(arg2_free_indices[end], get_next_letter())),
        )
    end

    throw(DomainError((arg1, arg2), "Multiplication with $arg1 and $arg2 is ambiguous"))
end

function +(arg1::TensorValue, arg2::TensorValue)
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

# TODO: This doesn't make sense. Make adjoint applicable only to MatrixSymbols.
struct Adjoint <: TensorValue
    expr::Any
end

function get_free_indices(arg::Adjoint)
    free_indices = get_free_indices(arg.expr)

    @assert length(free_indices) <= 2

    return reverse(LowerOrUpperIndex[i for i ∈ free_indices])
end

function adjoint(arg::UnaryOperation)
    return Adjoint(arg)
end

function adjoint(arg::BinaryOperation{*})
    free_ids = get_free_indices(arg)

    t = arg

    for i ∈ free_ids
        t = BinaryOperation{*}(t, KrD(flip(i), flip(i)))
    end

    return Adjoint(t)
end

function adjoint(arg::BinaryOperation{+})
    return arg.arg1' + arg.arg2'
end

function adjoint(arg::Union{Tensor,KrD,Zero})
    free_indices = get_free_indices(arg)

    if length(free_indices) > 2
        throw(DomainError(arg.id, "Adjoint is only defined for vectors and matrices"))
    end

    e = arg

    for i ∈ free_indices
        e = BinaryOperation{*}(e, KrD(flip(i), flip(i)))
    end

    return Adjoint(e)
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
        if d == 0
            push!(text, Char(0x2070))
        end
        if d == 1
            push!(text, Char(0x00B9))
        end
        if d == 2
            push!(text, Char(0x00B2))
        end
        if d == 3
            push!(text, Char(0x00B3))
        end
        if d > 3
            push!(text, Char(0x2070 + d))
        end
    end

    return join(text)
end

function to_string(arg::Tensor)
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

function to_string(arg::Sin)
    return "sin($(arg.arg))"
end

function to_string(arg::Cos)
    return "cos($(arg.arg))"
end

function parenthesize(arg)
    return to_string(arg)
end

function parenthesize(arg::BinaryOperation{+})
    return "(" * to_string(arg) * ")"
end

function to_string(arg::BinaryOperation{*})
    return parenthesize(arg.arg1) * parenthesize(arg.arg2)
end

function to_string(arg::Adjoint)
    return to_string(arg.expr)
end

function to_string(arg::BinaryOperation{+})
    return to_string(arg.arg1) * " + " * to_string(arg.arg2)
end

function Base.show(io::IO, expr::TensorValue)
    print(io, to_string(expr))
end
