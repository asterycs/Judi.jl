import Base.==
import Base.*
import Base.+
import Base.-
import Base.adjoint
import Base.broadcast
import Base.sin
import Base.cos
import Base.show

export Tensor, KrD, Zero

export tr

export equivalent

export to_string
export to_std_string

abstract type TensorValue end

# Shortcut for simpler comparison from
# https://stackoverflow.com/questions/62336686/struct-equality-with-arrays
function ==(a::T, b::T) where {T<:TensorValue}
    f = fieldnames(T)

    return (getfield.(Ref(a), f) == getfield.(Ref(b), f)) ||
           (reverse(getfield.(Ref(a), f)) == getfield.(Ref(b), f))
end

Value = Union{TensorValue,Real}

function _get_indices(arg::Real)
    return LowerOrUpperIndex[]
end

function equivalent(arg1::Real, arg2::Real)
    return arg1 == arg2
end

struct Tensor <: TensorValue
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

function can_remap(left::IndexList, right::IndexList)
    lu = unique(left)
    ru = unique(right)

    if length(lu) != length(ru)
        return false
    end

    let
        lu_letters = [i.letter for i ∈ left]
        ru_letters = [i.letter for i ∈ right]

        if length(unique(lu_letters)) != length(unique(ru_letters))
            return false
        end
    end

    index_map = Dict((l => r) for (l,r) ∈ zip(lu, ru))

    left_remap = [index_map[i] for i ∈ left]

    return left_remap == right
end

function equivalent(left, right)
    left_ids,right_ids = get_free_indices.((left, right))

    return can_remap(left_ids, right_ids)
end

function equivalent(left::Tensor, right::Tensor)
    left_ids,right_ids = get_free_indices.((left, right))

    return left.id == right.id && can_remap(left_ids, right_ids)
end

function are_unique(arg::AbstractArray)
    return length(unique(arg)) == length(arg)
end

struct KrD <: TensorValue
    indices::IndexList

    function KrD(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if !are_unique(indices) || length(indices) != 2
            throw(DomainError(indices, "Indices of δ are invalid"))
        end

        new(indices)
    end
end

function equivalent(left::KrD, right::KrD)
    return can_remap(left.indices, right.indices)
end

struct Zero <: TensorValue
    indices::IndexList

    function Zero(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
            throw(DomainError(indices, "Indices of 0 are invalid"))
        end

        new(indices)
    end
end

function equivalent(left::Zero, right::Zero)
    return can_remap(get_free_indices(left), get_free_indices(right))
end

struct BinaryOperation{Op} <: TensorValue where {Op}
    arg1::Value
    arg2::Value
end

abstract type AdditiveOperation end
struct Add <: AdditiveOperation end
struct Sub <: AdditiveOperation end
struct Mult end

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

UnaryValue = Union{Tensor,KrD,Zero,UnaryOperation,Real}

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

function eliminate_indices(arg1::IndexList, arg2::IndexList)
    return first(_eliminate_indices(arg1, arg2))
end

function eliminated_indices(arg1::IndexList, arg2::IndexList)
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

function _get_indices(arg::Union{Tensor,KrD,Zero})
    @assert length(unique(arg.indices)) == length(arg.indices)

    return arg.indices
end

function _get_indices(arg::Sin)
    return _get_indices(arg.arg)
end

function _get_indices(arg::Cos)
    return _get_indices(arg.arg)
end

function _get_indices(arg::Negate)
    return _get_indices(arg.arg)
end

function _get_indices(arg::BinaryOperation{Mult})
    return [_get_indices(arg.arg1); _get_indices(arg.arg2)]
end

function _get_indices(arg::BinaryOperation{Op}) where {Op}
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert is_permutation(arg1_ids, arg2_ids)

    return arg1_ids
end

function get_free_indices(arg)
    return eliminate_indices(_get_indices(evaluate(arg)))
end

function are_indices_unique(indices::IndexList)
    return length(unique(indices)) == length(indices)
end

function can_contract(arg1::Value, arg2::Value)
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

    if !isempty(pairs)
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

function is_valid_matrix_multiplication(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return typeof(flip(arg1_indices[end])) == typeof(arg2_indices[1])
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

    return BinaryOperation{Mult}(arg, KrD(flip(free_ids[2]), flip(free_ids[1])))
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

    return BinaryOperation{Mult}(new_arg1, arg2)
end

function *(arg1::TensorValue, arg2::Real)
    return arg2 * arg1
end

function *(arg1::Value, arg2::TensorValue)
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

    if !is_valid_matrix_multiplication(arg1, arg2)
        throw(DomainError((arg1, arg2), "Invalid matrix multiplication"))
    end

    intersecting_letters = intersect(get_letters(arg1_free_indices), get_letters(arg2_free_indices))

    if !isempty(intersecting_letters)
        if length(intersecting_letters) > 1 || arg1_free_indices[end].letter != arg2_free_indices[1].letter
            # Intersecting letters need updating if:
            #   - There are multiple intersecting letters
            #   - The intersecting letters are any other than the contracting indices

            for i ∈ arg1_free_indices
                if i.letter ∈ intersecting_letters
                    arg1 = update_index(arg1, i, same_to(i, get_next_letter()))
                end
            end
        end

        arg1_free_indices = get_free_indices(arg1)
    end


    if typeof(arg1_free_indices[end]) == Lower && typeof(arg2_free_indices[1]) == Upper
        return BinaryOperation{Mult}(
            update_index(arg1, arg1_free_indices[end], flip(arg2_free_indices[1])),
            arg2,
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
                same_to(arg2_free_indices[end], get_next_letter()),
            ),
        )
    end

    throw(DomainError((arg1, arg2), "Multiplication with $arg1 and $arg2 is ambiguous"))
end

function get_letters(indices::IndexList)
    return [i.letter for i ∈ indices]
end

function +(arg1::TensorValue, arg2::TensorValue)
    return create_additive_op(Add, arg1, arg2)
end

function -(arg1::TensorValue, arg2::TensorValue)
    return create_additive_op(Sub, arg1, arg2)
end

# TODO: Constrain op
function create_additive_op(op, arg1::TensorValue, arg2::TensorValue)
    arg1_ids, arg2_ids = get_free_indices.((arg1, arg2))

    if length(unique(arg1_ids)) != length(unique(arg2_ids))
        op_text = if op == +
            "add"
        elseif op == -
            "subtract"
        end

        throw(DomainError((arg1, arg2), "Cannot $op_text tensors of different order"))
    end

    if arg1_ids == arg2_ids
        return BinaryOperation{op}(arg1, arg2)
    end

    new_ids = [get_next_letter() for _ ∈ 1:length(unique(arg1_ids))]

    arg1_index_map = Dict((old => new for (old,new) ∈ zip(unique(arg1_ids), new_ids)))
    arg2_index_map = Dict((old => new for (old,new) ∈ zip(unique(arg2_ids), new_ids)))

    for index ∈ unique(arg1_ids)
        arg1 = BinaryOperation{Mult}(arg1, KrD(flip(index), same_to(index, arg1_index_map[index])))
    end

    for index ∈ union(arg2_ids)
        arg2 = BinaryOperation{Mult}(arg2, KrD(flip(index), same_to(index, arg2_index_map[index])))
    end

    return BinaryOperation{op}(arg1, arg2)
end

function update_index(arg::TensorValue, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    indices = get_free_indices(arg)

    if from == to
        return arg
    end

    @assert from ∈ indices
    @assert typeof(from) != typeof(flip(to)) "update_index shall not transpose"

    return BinaryOperation{Mult}(arg, KrD(flip(from), to))
end

function reshape(arg::TensorValue, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    indices = get_free_indices(arg)

    if from == to
        return arg
    end

    @assert from ∈ indices

    return BinaryOperation{Mult}(arg, KrD(flip(from), to))
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

function adjoint(arg::BinaryOperation{Mult})
    free_ids = get_free_indices(arg)

    t = arg

    for i ∈ union(free_ids)
        t = BinaryOperation{Mult}(t, KrD(flip(i), flip_to(i, get_next_letter())))
    end

    return Adjoint(t)
end

function adjoint(arg::BinaryOperation{Op}) where {Op}
    arg1_ids = get_free_indices(arg.arg1)
    arg2_ids = get_free_indices(arg.arg2)

    @assert length(unique(arg1_ids)) == length(unique(arg2_ids))

    new_ids = [get_next_letter() for _ ∈ 1:length(unique(arg1_ids))]

    arg1_index_map = Dict((old => new for (old,new) ∈ zip(unique(arg1_ids), new_ids)))
    arg2_index_map = Dict((old => new for (old,new) ∈ zip(unique(arg2_ids), new_ids)))

    arg1_t = arg.arg1
    arg2_t = arg.arg2

    for index ∈ unique(arg1_ids)
        arg1_t = BinaryOperation{Mult}(arg1_t, KrD(flip(index), flip_to(index, arg1_index_map[index])))
    end

    for index ∈ union(arg2_ids)
        arg2_t = BinaryOperation{Mult}(arg2_t, KrD(flip(index), flip_to(index, arg2_index_map[index])))
    end

    return Adjoint(BinaryOperation{Op}(arg1_t, arg2_t))
end

function adjoint(arg::Union{Tensor,KrD,Zero})
    free_indices = get_free_indices(arg)

    if length(free_indices) > 2
        throw(DomainError(arg.id, "Adjoint is only defined for vectors and matrices"))
    end

    e = arg

    for i ∈ free_indices
        e = BinaryOperation{Mult}(e, KrD(flip(i), flip_to(i, get_next_letter())))
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

function to_string(arg::Adjoint)
    return to_string(arg.expr)
end

function to_string(arg::BinaryOperation{Add})
    return to_string(arg.arg1) * " + " * to_string(arg.arg2)
end

function to_string(arg::BinaryOperation{Sub})
    return to_string(arg.arg1) * " - " * parenthesize(arg.arg2)
end

function Base.show(io::IO, expr::TensorValue)
    return print(io, to_string(expr))
end
