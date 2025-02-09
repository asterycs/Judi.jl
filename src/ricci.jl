# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import Base.==
import Base.*
import Base.+
import Base.-
import Base.adjoint
import Base.broadcast
import Base.sin
import Base.cos
import Base.show

export tr

abstract type TensorExpr end

# Shortcut for simpler comparison from
# https://stackoverflow.com/questions/62336686/struct-equality-with-arrays
function ==(a::T, b::T) where {T<:TensorExpr}
    f = fieldnames(T)

    return (getfield.(Ref(a), f) == getfield.(Ref(b), f)) ||
           (reverse(getfield.(Ref(a), f)) == getfield.(Ref(b), f))
end

Value = Union{TensorExpr,Real}

function get_indices(arg::Real)
    return LowerOrUpperIndex[]
end

struct Tensor <: TensorExpr
    id::String
    indices::IndexList

    function Tensor(id, indices::LowerOrUpperIndex...)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
            throw(DomainError(indices, "Indices of $id are invalid"))
        end

        new(id, indices)
    end
end

function are_unique(arg::AbstractArray)
    return length(unique(arg)) == length(arg)
end

struct KrD <: TensorExpr
    indices::IndexList

    function KrD(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        # Regarding !isempty(eliminated_indices(indices)):
        # A contraction here (or trace) is not semantically wrong. However, it makes
        # it more difficult to identify traces (and vector sums) when evaluating.
        if !are_unique(indices) ||
           length(indices) != 2 ||
           !isempty(eliminated_indices(indices))
            throw(DomainError(indices, "Indices of δ are invalid"))
        end

        new(indices)
    end
end

struct Zero <: TensorExpr
    indices::IndexList

    function Zero(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
            throw(DomainError(indices, "Indices of 0 are invalid"))
        end

        new(indices)
    end
end

struct BinaryOperation{Op} <: TensorExpr where {Op}
    arg1::Value
    arg2::Value
end

abstract type AdditiveOperation end
struct Add <: AdditiveOperation end
struct Sub <: AdditiveOperation end
struct Mult end

abstract type UnaryOperation <: TensorExpr end

UnaryValue = Union{Tensor,KrD,Zero,UnaryOperation,Real}

struct Negate <: UnaryOperation
    arg::TensorExpr
end

struct Sin <: UnaryOperation
    arg::TensorExpr
end

function sin(arg::TensorExpr)
    return Sin(arg)
end

struct Cos <: UnaryOperation
    arg::TensorExpr
end

function cos(arg::TensorExpr)
    return Cos(arg)
end

function _eliminate_indices(arg1::IndexList, arg2::IndexList)
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

function _eliminate_indices(arg::IndexList)
    CanBeNothing = Union{Nothing,Lower,Upper}
    available = CanBeNothing[i for i ∈ unique(arg)]
    eliminated = LowerOrUpperIndex[]

    for i ∈ eachindex(available)
        if isnothing(available[i])
            continue
        end

        for j ∈ eachindex(available)
            if isnothing(available[j])
                continue
            end

            if flip(available[j]) == available[i] # contraction
                push!(eliminated, available[i])
                push!(eliminated, available[j])
                available[i] = nothing
                available[j] = nothing
            end
        end
    end

    filtered = filter(i -> i ∈ available, arg)

    return filtered, eliminated
end

function eliminate_indices(arg::IndexList)
    return first(_eliminate_indices(arg))
end

function eliminated_indices(arg::IndexList)
    remaining = first(_eliminate_indices(arg))
    return setdiff(arg, remaining)
end

function eliminate_indices(arg1::IndexList, arg2::IndexList)
    return first(_eliminate_indices(arg1, arg2))
end

function eliminated_indices(arg1::IndexList, arg2::IndexList)
    return last(_eliminate_indices(arg1, arg2))
end

function count_values(input::AbstractArray{T}) where {T}
    return Dict((i => count(==(i), input)) for i ∈ unique(input))
end

function get_next_letter(exprs...)
    exprs = [e for e ∈ exprs]
    indices = get_indices.(exprs)

    letters = [index.letter for index ∈ Iterators.flatten(indices)]
    max_letter = maximum(letters)

    return max_letter + 1
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

function is_permutation(arg1::TensorExpr, arg2::TensorExpr)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return is_permutation(unique(arg1_indices), unique(arg2_indices))
end

function get_indices(arg::Union{Tensor,KrD,Zero})
    @assert length(unique(arg.indices)) == length(arg.indices)

    return arg.indices
end

function get_indices(arg::Sin)
    return get_indices(arg.arg)
end

function get_indices(arg::Cos)
    return get_indices(arg.arg)
end

function get_indices(arg::Negate)
    return get_indices(arg.arg)
end

function get_indices(arg::BinaryOperation{Mult})
    return [get_indices(arg.arg1); get_indices(arg.arg2)]
end

function get_indices(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    arg1_free_ids, arg2_free_ids = get_free_indices.((arg.arg1, arg.arg2))

    @assert is_permutation(arg1_free_ids, arg2_free_ids)

    arg1_ids, arg2_ids = get_indices.((arg.arg1, arg.arg2))

    return union(arg1_ids, arg2_ids)
end

function get_free_indices(arg)
    return unique(eliminate_indices(get_indices(evaluate(arg))))
end

function are_indices_unique(indices::IndexList)
    return length(unique(indices)) == length(indices)
end

function can_contract(arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    # If there is at least one matching index pair then the contraction is unambigous.
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

    if !isempty(pairs)
        return true
    end

    return false
end

function tr(arg::TensorExpr)
    free_ids = get_free_indices(arg)

    de = DomainError("Trace is defined only for matrices")

    if length(free_ids) != 2
        throw(de)
    end

    if all(typeof.(free_ids) .== Upper) || all(typeof.(free_ids) .== Lower)
        throw(de)
    end

    return BinaryOperation{Mult}(arg, KrD(flip(free_ids[2]), flip(free_ids[1])))
end

function Base.broadcasted(::typeof(*), arg1::TensorExpr, arg2::TensorExpr)
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

    return BinaryOperation{Mult}(new_arg1, arg2)
end

function *(arg1::TensorExpr, arg2::Real)
    return arg2 * arg1
end

function *(arg1::Value, arg2::TensorExpr)
    arg1_free_indices = get_free_indices(arg1)
    arg2_free_indices = get_free_indices(arg2)

    if isempty(arg1_free_indices) || isempty(arg2_free_indices)
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if length(arg1_free_indices) > 2
        throw(DomainError(arg1, "Multiplication involving tensor \"$arg1\" is ambiguous"))
    end

    if length(arg2_free_indices) > 2
        throw(DomainError(arg2, "Multiplication involving tensor \"$arg2\" is ambiguous"))
    end

    intersecting_letters =
        intersect(get_letters(arg1_free_indices), get_letters(arg2_free_indices))

    if !isempty(intersecting_letters)
        if length(intersecting_letters) > 1 ||
           arg1_free_indices[end].letter != arg2_free_indices[1].letter
            # Intersecting letters need updating if:
            #   - There are multiple intersecting letters
            #   - The intersecting letters are any other than the contracting indices

            for i ∈ arg1_free_indices
                if i.letter ∈ intersecting_letters
                    arg1 = update_index(arg1, i, same_to(i, get_next_letter(arg1, arg2)))
                end
            end
        end

        arg1_free_indices = get_free_indices(arg1)
    end

    # TODO: WETWET, simplify, add e.g. get_lower(arg::IndexList) and get_upper(arg::IndexLists)
    if typeof(arg1_free_indices[end]) == Lower && typeof(arg2_free_indices[1]) == Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(
                arg1,
                arg1_free_indices[end],
                same_to(arg1_free_indices[end], new_letter),
            ),
            update_index(
                arg2,
                arg2_free_indices[1],
                same_to(arg2_free_indices[1], new_letter),
            ),
        )
    end

    if typeof(arg1_free_indices[1]) == Lower && typeof(arg2_free_indices[1]) == Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(
                arg1,
                arg1_free_indices[1],
                same_to(arg1_free_indices[1], new_letter),
            ),
            update_index(
                arg2,
                arg2_free_indices[1],
                same_to(arg2_free_indices[1], new_letter),
            ),
        )
    end

    if typeof(arg1_free_indices[1]) == Lower && typeof(arg2_free_indices[end]) == Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(
                arg1,
                arg1_free_indices[1],
                same_to(arg1_free_indices[1], new_letter),
            ),
            update_index(
                arg2,
                arg2_free_indices[end],
                same_to(arg2_free_indices[end], new_letter),
            ),
        )
    end

    if typeof(arg1_free_indices[end]) == Lower && typeof(arg2_free_indices[end]) == Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(
                arg1,
                arg1_free_indices[end],
                same_to(arg1_free_indices[end], new_letter),
            ),
            update_index(
                arg2,
                arg2_free_indices[end],
                same_to(arg2_free_indices[end], new_letter),
            ),
        )
    end

    if length(arg1_free_indices) == 1 &&
       length(arg2_free_indices) == 1 &&
       typeof(arg1_free_indices[end]) == Upper &&
       typeof(arg2_free_indices[1]) == Lower
        return BinaryOperation{Mult}(
            arg1,
            update_index(
                arg2,
                arg2_free_indices[end],
                same_to(arg2_free_indices[end], get_next_letter(arg1, arg2)),
            ),
        )
    end

    throw(DomainError((arg1, arg2), "Multiplication with $arg1 and $arg2 is ambiguous"))
end

function get_letters(indices::IndexList)
    return [i.letter for i ∈ indices]
end

function +(arg1::TensorExpr, arg2::TensorExpr)
    return create_additive_op(Add(), arg1, arg2)
end

function -(arg1::TensorExpr, arg2::TensorExpr)
    return create_additive_op(Sub(), arg1, arg2)
end

function create_additive_op(
    op::Op,
    arg1::TensorExpr,
    arg2::TensorExpr,
) where {Op<:AdditiveOperation}
    arg1_ids, arg2_ids = get_free_indices.((arg1, arg2))

    if length(unique(arg1_ids)) != length(unique(arg2_ids))
        op_text = if typeof(Op) == Add
            "add"
        elseif typeof(Op) == Sub
            "subtract"
        end

        throw(DomainError((arg1, arg2), "Cannot $op_text tensors of different order"))
    end

    if arg1_ids == arg2_ids
        return BinaryOperation{Op}(arg1, arg2)
    end

    next_letter = get_next_letter(arg1, arg2)

    new_ids = [next_letter + i - 1 for i ∈ 1:length(unique(arg1_ids))]

    arg1_index_map = Dict((old => new for (old, new) ∈ zip(unique(arg1_ids), new_ids)))
    arg2_index_map = Dict((old => new for (old, new) ∈ zip(unique(arg2_ids), new_ids)))

    for index ∈ unique(arg1_ids)
        arg1 = BinaryOperation{Mult}(
            arg1,
            KrD(flip(index), same_to(index, arg1_index_map[index])),
        )
    end

    for index ∈ union(arg2_ids)
        arg2 = BinaryOperation{Mult}(
            arg2,
            KrD(flip(index), same_to(index, arg2_index_map[index])),
        )
    end

    return BinaryOperation{Op}(arg1, arg2)
end

function update_index(
    arg::TensorExpr,
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex;
    allow_shape_change = false,
)
    indices = get_free_indices(arg)

    if from == to
        return arg
    end

    @assert from ∈ indices

    if !allow_shape_change
        if typeof(from) != typeof(to)
            throw(DomainError("A shape change is not permitted"))
        end
    end

    return BinaryOperation{Mult}(arg, KrD(flip(from), to))
end

function -(arg::TensorExpr)
    return Negate(arg)
end

function adjoint(arg::T) where {T<:UnaryOperation}
    return T(arg.arg')
end

function adjoint(arg::BinaryOperation{Mult})
    return BinaryOperation{Mult}(adjoint(arg.arg2), adjoint(arg.arg1))
end

function adjoint(arg::BinaryOperation{Op}) where {Op}
    arg1_ids = unique(get_free_indices(arg.arg1))
    arg2_ids = unique(get_free_indices(arg.arg2))

    @assert length(unique(arg1_ids)) == length(unique(arg2_ids))

    arg1_t = arg.arg1
    arg2_t = arg.arg2

    for i ∈ arg1_ids
        tmp_letter = get_next_letter(arg1_t, arg2_t)
        arg1_t = BinaryOperation{Mult}(
            BinaryOperation{Mult}(arg1_t, KrD(flip(i), flip_to(i, tmp_letter))),
            KrD(same_to(i, tmp_letter), flip(i)),
        )
    end

    for i ∈ arg2_ids
        tmp_letter = get_next_letter(arg1_t, arg2_t)
        arg2_t = BinaryOperation{Mult}(
            BinaryOperation{Mult}(arg2_t, KrD(flip(i), flip_to(i, tmp_letter))),
            KrD(same_to(i, tmp_letter), flip(i)),
        )
    end

    return BinaryOperation{Op}(arg1_t, arg2_t)
end

function adjoint(arg::Union{Tensor,KrD,Zero})
    free_indices = unique(get_free_indices(arg))

    if length(free_indices) > 2
        throw(DomainError(arg.id, "Adjoint is only defined for vectors and matrices"))
    end

    e = arg

    for i ∈ free_indices
        tmp_letter = get_next_letter(e)
        e = BinaryOperation{Mult}(
            BinaryOperation{Mult}(e, KrD(flip(i), flip_to(i, tmp_letter))),
            KrD(same_to(i, tmp_letter), flip(i)),
        )
    end

    return e
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
    return "-$(parenthesize(arg.arg))"
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

function parenthesize(arg::BinaryOperation{Add})
    return "(" * to_string(arg) * ")"
end

function parenthesize(arg::BinaryOperation{Sub})
    return "(" * to_string(arg) * ")"
end

function to_string(arg::BinaryOperation{Mult})
    return parenthesize(arg.arg1) * parenthesize(arg.arg2)
end

function to_string(arg::BinaryOperation{Add})
    return to_string(arg.arg1) * " + " * to_string(arg.arg2)
end

function to_string(arg::BinaryOperation{Sub})
    return to_string(arg.arg1) * " - " * parenthesize(arg.arg2)
end

function Base.show(io::IO, expr::TensorExpr)
    return print(io, to_string(expr))
end
