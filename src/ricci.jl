import Base.==
import Base.*
import Base.+
import Base.-
import Base.adjoint
import Base.broadcast
import Base.hash
import Base.sin
import Base.cos
import Base.show

export Upper, Lower
export Tensor, KrD, Zero

export tr

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

function flip_to(index::Lower, letter::Letter)
    return Upper(letter)
end

function flip_to(index::Upper, letter::Letter)
    return Lower(letter)
end

function same_to(old::Lower, letter::Letter)
    return Lower(letter)
end

function same_to(old::Upper, letter::Letter)
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

# TODO: Find a better solution for this. Perhaps a set with taken indices?
const global NEXT_LETTER = Ref{Letter}(100)

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

Value = Union{TensorValue,Real}
IndexSet = Vector{LowerOrUpperIndex}

function get_free_indices(arg::Real)
    return LowerOrUpperIndex[]
end

function equivalent(arg1::Real, arg2::Real)
    return arg1 == arg2
end

struct Tensor <: TensorValue
    id::String
    indices::IndexSet

    function Tensor(id, indices::LowerOrUpperIndex...)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
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

        if length(unique(indices)) != length(indices) || length(indices) != 2
            throw(DomainError(indices, "Indices of δ are invalid"))
        end

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

        if length(unique(indices)) != length(indices)
            throw(DomainError(indices, "Indices of 0 are invalid"))
        end

        new(indices)
    end
end

function equivalent(left::Zero, right::Zero)
    return all(typeof.(left.indices) .== typeof.(right.indices))
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

struct Negate <: UnaryOperation
    arg::TensorValue
end

struct Sin <: UnaryOperation
    arg::TensorValue
end

function sin(arg::TensorValue)
    return Sin(arg)
end

struct Cos <: UnaryOperation
    arg::TensorValue
end

function cos(arg::TensorValue)
    return Cos(arg)
end

NonTrivialValue = Union{Tensor,KrD,BinaryOperation{*},BinaryOperation{+},Real}
# TODO: Rename BinaryOperation{*} and align with Mult below
NonTrivialNonMult = Union{Tensor,KrD,BinaryOperation{+},Real}

function _eliminate_indices(arg1::IndexSet, arg2::IndexSet)
    CanBeNothing = Union{Nothing,Lower,Upper}
    available1 = CanBeNothing[i for i ∈ unique(arg1)]
    available2 = CanBeNothing[i for i ∈ unique(arg2)]
    eliminated = LowerOrUpperIndex[]

    for i ∈ eachindex(available1)
        if isnothing(available1[i])
            continue
        end

        for j ∈ eachindex(available2)
            if isnothing(available2[j])
                continue
            end

            if flip(available2[j]) == available1[i] # contraction
                push!(eliminated, available1[i])
                push!(eliminated, available2[j])
                available1[i] = nothing
                available2[j] = nothing
            end
        end
    end

    filtered1 = filter(i -> i ∈ available1, arg1)
    filtered2 = filter(i -> i ∈ available2, arg2)

    return (filtered1, filtered2), eliminated
end

function eliminate_indices(arg1::IndexSet, arg2::IndexSet)
    return first(_eliminate_indices(arg1, arg2))
end

function eliminated_indices(arg1::IndexSet, arg2::IndexSet)
    return last(_eliminate_indices(arg1, arg2))
end

function count_values(input::AbstractArray{T}) where {T}
    return Dict((i => count(==(i), input)) for i ∈ unique(input))
end

function is_permutation(l::AbstractArray{T}, r::AbstractArray{T}) where {T}
    if length(l) != length(r)
        return false
    end

    l_element_count = count_values(l)
    r_element_count = count_values(r)

    for index ∈ keys(l_element_count)
        if l_element_count[index] != r_element_count[index]
            return false
        end
    end

    return true
end

function is_permutation(arg1::TensorValue, arg2::TensorValue)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return is_permutation(unique(arg1_indices), unique(arg2_indices))
end

function get_free_indices(arg::Union{Tensor,KrD,Zero})
    @assert length(unique(arg.indices)) == length(arg.indices)

    return arg.indices
end

function get_free_indices(arg::Sin)
    return get_free_indices(arg.arg)
end

function get_free_indices(arg::Cos)
    return get_free_indices(arg.arg)
end

function get_free_indices(arg::Negate)
    return get_free_indices(arg.arg)
end

function get_free_indices(arg::BinaryOperation{*})
    arg1_free_indices, arg2_free_indices =
        eliminate_indices(get_free_indices(arg.arg1), get_free_indices(arg.arg2))

    return [arg1_free_indices; arg2_free_indices]
end

function get_free_indices(arg::BinaryOperation{Op}) where {Op}
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert is_permutation(typeof.(arg1_ids), typeof.(arg2_ids))

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

function can_contract(arg1::TensorValue, arg2::TensorValue)
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
# However, arg1.indices[end] and arg2.indices[1] letters do not
# have to match since they can be updated.
function is_valid_matrix_multiplication(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    for (i,l) ∈ enumerate(arg1_indices)
        for (j,r) ∈ enumerate(arg2_indices)
            if l.letter == r.letter && (i != length(arg1_indices) || j != 1)
                return false
            end
        end
    end

    if typeof(flip(arg1_indices[end])) != typeof(arg2_indices[1])
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

function Base.broadcasted(::typeof(*), arg1::TensorValue, arg2::TensorValue)
    arg1_free_indices = get_free_indices(arg1)
    arg2_free_indices = get_free_indices(arg2)

    if length(arg1_free_indices) != length(arg2_free_indices)
        return throw(
            DomainError(
                (arg1, arg2),
                "Cannot do elementwise multiplication with inputs of different order",
            ),
        )
    end

    if typeof.(arg1_free_indices) != typeof.(arg2_free_indices)
        return throw(
            DomainError(
                (arg1, arg2),
                "Elementwise multiplication is ambiguous for tensors with different co/contravariance",
            ),
        )
    end

    new_arg1 = arg1

    for (li, ri) ∈ zip(arg1_free_indices, arg2_free_indices)
        new_arg1 = update_index(new_arg1, li, ri)
    end

    return BinaryOperation{*}(new_arg1, arg2)
end

function *(arg1::TensorValue, arg2::Real)
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

    if !is_valid_matrix_multiplication(arg1, arg2)
        throw(DomainError((arg1, arg2), "Invalid matrix multiplication"))
    end

    if typeof(arg1_free_indices[end]) == Lower && typeof(arg2_free_indices[1]) == Upper
        return BinaryOperation{*}(
            update_index(arg1, arg1_free_indices[end], flip(arg2_free_indices[1])),
            arg2,
        )
    end

    if length(arg1_free_indices) == 1 &&
       length(arg2_free_indices) == 1 &&
       typeof(arg1_free_indices[end]) == Upper &&
       typeof(arg2_free_indices[1]) == Lower
        return BinaryOperation{*}(
            arg1,
            update_index(
                arg2,
                arg2_free_indices[end],
                same_to(arg2_free_indices[end], get_next_letter()),
            ),
        )
    end

    throw(DomainError((arg1, arg2), "Multiplication with $arg1 and $arg2 is ambiguous"))
end

function +(arg1::TensorValue, arg2::TensorValue)
    BinaryOperation{+}(arg1, arg2)
end

function -(arg1::TensorValue, arg2::TensorValue)
    BinaryOperation{-}(arg1, arg2)
end

function update_index(arg::TensorValue, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    indices = get_free_indices(arg)

    if from == to
        return arg
    end

    @assert from ∈ indices
    @assert typeof(from) != typeof(flip(to)) "update_index shall not transpose"

    return BinaryOperation{*}(arg, KrD(flip(from), to))
end

function -(arg::TensorValue)
    return Negate(arg)
end

struct Adjoint <: TensorValue
    expr::TensorValue
end

function get_free_indices(arg::Adjoint)
    free_indices = get_free_indices(arg.expr)

    @assert length(free_indices) <= 2

    return reverse(LowerOrUpperIndex[i for i ∈ free_indices])
end

function adjoint(arg::T) where T <: UnaryOperation
    return T(arg.arg')
end

function adjoint(arg::BinaryOperation{*})
    free_ids = get_free_indices(arg)

    t = arg

    for i ∈ union(free_ids)
        t = BinaryOperation{*}(t, KrD(flip(i), flip_to(i, get_next_letter())))
    end

    return Adjoint(t)
end

function adjoint(arg::BinaryOperation{Op}) where {Op}
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert length(arg1_ids) == length(arg2_ids)

    arg1_t = arg.arg1
    arg2_t = arg.arg2

    new_ids = [get_next_letter() for _ ∈ 1:length(union(arg1_ids))]

    for (i,index) ∈ enumerate(union(arg1_ids))
        arg1_t = BinaryOperation{*}(arg1_t, KrD(flip(index), flip_to(index, new_ids[i])))
    end

    for (i,index) ∈ enumerate(union(arg2_ids))
        arg2_t = BinaryOperation{*}(arg2_t, KrD(flip(index), flip_to(index, new_ids[i])))
    end

    return Adjoint(Op(arg1_t, arg2_t))
end

function adjoint(arg::Union{Tensor,KrD,Zero})
    free_indices = get_free_indices(arg)

    if length(free_indices) > 2
        throw(DomainError(arg.id, "Adjoint is only defined for vectors and matrices"))
    end

    e = arg

    for i ∈ free_indices
        e = BinaryOperation{*}(e, KrD(flip(i), flip_to(i, get_next_letter())))
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

    return arg.id * join(scripts)
end

function to_string(arg::KrD)
    scripts = [script(i) for i ∈ arg.indices]

    return "δ" * join(scripts)
end

function to_string(arg::Real)
    return string(arg)
end

function to_string(arg::Zero)
    scripts = [script(i) for i ∈ arg.indices]

    return "0" * join(scripts)
end

function to_string(arg::Negate)
    return "-$(arg.arg)"
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

function parenthesize(arg::BinaryOperation{-})
    return "(" * to_string(arg) * ")"
end

function to_string(arg::BinaryOperation{*})
    return parenthesize(arg.arg1) * parenthesize(arg.arg2)
end

function to_string(arg::Adjoint)
    return to_string(arg.expr)
end

function to_string(arg::BinaryOperation{Op}) where {Op}
    return to_string(arg.arg1) * " " * string(Op) * " " * to_string(arg.arg2)
end

function Base.show(io::IO, expr::TensorValue)
    return print(io, to_string(expr))
end
