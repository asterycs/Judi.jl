using Revise

import Base.*
import Base.+
import Base.adjoint
import Base.:(==)

export Upper, Lower
export Sym

export record
export assemble_pullback
export simplify
export to_string

Letter = Int64

struct Upper
    letter::Letter
end

struct Lower
    letter::Letter
end

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

abstract type SymbolicValue end

LowerOrUpperIndex = Union{Lower, Upper}
IndexSet = Vector{LowerOrUpperIndex}

struct Sym <: SymbolicValue
    id::String
    indices::IndexSet
end

struct KrD <: SymbolicValue
    indices::IndexSet
end

struct Zero <: SymbolicValue
    indices::IndexSet
end

struct Transpose <: SymbolicValue end

mutable struct BinaryOperation <: SymbolicValue
    op
    arg1
    arg2
end

function get_indices(arg::Sym)
    arg.indices
end

function get_indices(arg::BinaryOperation)
    unique([get_indices(arg.arg1); get_indices(arg.arg2)])
end

# # TODO: Make the check on the matrix stronger.
# # Should check that the indices contains one Lower and one Upper,
# function is_contraction_valid(arg1::BinaryOperation, arg2::Sym)
#     if length(arg1.indices) == 2
#         if length(arg2.indices) == 1
#             return typeof(arg2.indices[1]) == Upper
#         else
#             @assert false "Not implemented"
#         end
#     elseif length(arg1.indices) == 1
#         if length(arg2.indices) == 2
#             return typeof(arg1.indices[1]) == Lower
#         else
#             @assert false "Not implemented"
#         end
#     else
#         @assert false "Not implemented"
#     end
# end

function has_index_type(arg, T)
    indices = get_indices(arg)

    for index ∈ indices
        if typeof(index) == T
            return true
        end
    end

    return false
end

function are_dimensions_valid(arg1, arg2)
    if isempty(get_indices(arg1)) && isempty(get_indices(arg2))
        return true
    else
        return has_index_type(arg1, Lower) && has_index_type(arg2, Upper)
    end
end

function *(arg1::SymbolicValue, arg2::SymbolicValue)
    @assert are_dimensions_valid(simplify(arg1), simplify(arg2))
    if isempty(get_indices(arg1)) || isempty(get_indices(arg2))
        return BinaryOperation(*, arg1, arg2)
    else
        return BinaryOperation(*, update_index(arg1, flip(arg2.indices[1])), arg2)
    end
end

function lower(indices::IndexSet)
    [Lower(idx.letter) for idx ∈ indices]
end

# Can assume that the index sets only have one letter in common
function is_valid_contraction(arg1::IndexSet, arg2::IndexSet, index::LowerOrUpperIndex)
    left_indices = Set(arg1)
    right_indices = Set(arg2)

    if index ∈ left_indices && flip(index) ∈ right_indices
        return true
    elseif flip(index) ∈ left_indices && index ∈ right_indices
        return true
    end

    return false
end

function replace!(indices::IndexSet, old::Letter, new::LowerOrUpperIndex)
    @assert count(i -> i.letter == old, indices) == 1

    for i ∈ eachindex(indices)
        if indices[i].letter == old
            indices[i] = new
        end
    end
end

function +(arg1::SymbolicValue, arg2::SymbolicValue)
    BinaryOperation(+, arg1, arg2)
end

function update_index(arg::Sym, index::LowerOrUpperIndex)
    println("Update index for Sym")
    if arg.indices[end] == index
        return arg
    else
        return BinaryOperation(*, arg, KrD([flip(arg.indices[end]); index]))
    end
end

function update_index(arg::BinaryOperation, index::LowerOrUpperIndex)
    s = simplify(arg)

    indices = get_indices(s)

    if indices[end] == index
        arg
    else
        BinaryOperation(*, arg, KrD([flip(indices[end]); index]))
    end
end

function adjoint(arg::SymbolicValue)
    BinaryOperation(*, arg, Transpose())
end

function ==(l::Sym, r::Sym)
    return l.id == r.id && l.indices == r.indices
end

function simplify(sym::Sym)
    sym
end

function evaluate(*, arg1::Sym, arg2::Transpose)
    newsym = deepcopy(arg1)

    for i ∈ eachindex(newsym.indices)
        newsym.indices[i] = flip(newsym.indices[i])
    end

    newsym
end

function evaluate(*, arg1::BinaryOperation, arg2::Transpose)
    BinaryOperation(*, arg1, arg2)
end

function evaluate(*, arg1::BinaryOperation, arg2::BinaryOperation)
    BinaryOperation(*, arg1, arg2)
end

function evaluate(*, arg1::Sym, arg2::KrD)
    println("sym times KrD")
    @show arg1
    @show arg2
    contracting_index = lower(arg1.indices) ∩ lower(arg2.indices)
    @show contracting_index
    @assert length(contracting_index) == 1

    contracting_index = contracting_index[1]

    @assert is_valid_contraction(arg1.indices, arg2.indices, contracting_index)

    @assert length(arg2.indices) == 2

    newarg = deepcopy(arg1)

    for i ∈ eachindex(newarg.indices)
        if newarg.indices[i].letter == arg2.indices[1].letter
            newarg.indices[i] = arg2.indices[2]
        end
    end

    newarg
end

function evaluate(*, arg1::BinaryOperation, arg2::KrD)
    BinaryOperation(*, arg1, arg2)
end

function evaluate(*, arg1::BinaryOperation, arg2::Sym)
    BinaryOperation(*, arg1, arg2)
end

function evaluate(*, arg1::Sym, arg2::Sym)
    BinaryOperation(*, arg1, arg2)
end

function evaluate(+, arg1::Zero, arg2)
    arg2
end

function evaluate(+, arg1, arg2::Zero)
    arg1
end

function simplify(op::BinaryOperation)
    evaluate(op.op, simplify(op.arg1), simplify(op.arg2))
end

# TODO: Rename all simplify to evaluate
function simplify(sym::Union{Sym, KrD, Transpose, Zero})
    sym
end

function to_string(arg::Sym)
    upper_indices = [i.letter for i ∈ arg.indices if typeof(i) == Upper]
    upper_indices = string(upper_indices...)
    lower_indices = [i.letter for i ∈ arg.indices if typeof(i) == Lower]
    lower_indices = string(lower_indices...)

    arg.id * "^(" * upper_indices * ")_(" * lower_indices * ")"
end

function to_string(arg::KrD)
    upper_indices = [i.letter for i ∈ arg.indices if typeof(i) == Upper]
    upper_indices = string(upper_indices...)
    lower_indices = [i.letter for i ∈ arg.indices if typeof(i) == Lower]
    lower_indices = string(lower_indices...)

    "δ" * "^(" * upper_indices * ")_(" * lower_indices * ")"
end

function to_string(arg::BinaryOperation)
    "(" * to_string(arg.arg1) * " " * string(*) * " " * to_string(arg.arg2) * ")"
end

function record(expr::SymbolicValue)
    op_index = 0
    graph = []

    function track(arg::Union{Sym, KrD, Transpose})
        for (index, (value, _)) ∈ enumerate(graph)
            if arg == value
                return index
            end
        end

        op_index += 1
        index = op_index

        push!(graph, (arg, index))

        index
    end

    function track(arg::BinaryOperation)
        i1 = track(arg.arg1)
        i2 = track(arg.arg2)

        op_index += 1
        index = op_index

        push!(graph, (arg.op, (i1, i2)))

        index
    end

    track(expr)

    graph
end



function backprop_rule(::typeof(Sym), arg)
    z = arg

    pullback = z_cot -> z_cot

    z, pullback
end

function backprop_rule(::typeof(*), arg1, arg2)
    z = evaluate(*, arg1, arg2)

    pullback = z_cot -> (evaluate(*, z_cot, arg2), evaluate(*, z_cot, arg1))

    z, pullback
end

function backprop_rule(::typeof(+), arg1, arg2)
    z = evaluate(+, arg1, arg2)

    pullback = z_cot -> (z_cot, z_cot)

    z, pullback
end


function assemble_pullback(graph)
    values = Any[typeof(v[1]) ∈ [Sym, KrD, Transpose] ? v[1] : nothing for v in graph]
    pullback_stack = []

    for (value_index, (operation, indices)) ∈ enumerate(graph)
        if typeof(operation) == Sym
            continue
        end
        if typeof(operation) == KrD
            continue
        end
        if typeof(operation) == Transpose
            continue
        end
        println(operation, " ", indices)
        @show values
        args = [values[index] for index in indices]

        value, pullback = backprop_rule(operation, args...)

        values[value_index] = value
        push!(pullback_stack, (pullback, indices))
    end

    @show values
    @show pullback_stack

    function pullback(output_cotangent)
        cotangent_values = Vector{Any}(nothing, length(values))
        fill!(cotangent_values, Zero([]))
        cotangent_values[end] = output_cotangent

        for (i, (pullback_fn, indices)) in enumerate(Iterators.reverse(pullback_stack))
            current_cotangent_value = cotangent_values[end - i + 1]
            cotangent_args = pullback_fn(current_cotangent_value)
            for (index, cotangent) in zip(indices, cotangent_args)
                cotangent_values[index] += cotangent
            end
        end
        
        cotangent_values[1:length(values)]
    end

    pullback
end

x = Sym("x", [Upper(3)])
A = Sym("A", [Upper(1); Lower(2)])

graph = record(A * x)
p = assemble_pullback(graph)

p(Sym("I", []))