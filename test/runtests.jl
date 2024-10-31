using MatrixCalculus
using Test

MC = MatrixCalculus

# @testset "End-to-end simple" begin
#     x = Sym("x", [Upper(3)])
#     A = Sym("A", [Upper(1); Lower(2)])

#     graph = record(A * x)
#     p = assemble_pullback(graph)

#     @test to_string.(simplify.(p(Sym("I", [])))) == "A"


#     graph = record(x' * A)
#     p = assemble_pullback(graph)

#     @test to_string.(simplify.(p(Sym("I", [])))) == "A"
# end

# @testset "Invalid product" begin
#     x = Sym("x", [Upper(3)])
#     A = Sym("A", [Upper(1); Lower(2)])

#     @test_throws DomainError x * A
# end

@testset "index equality operator" begin
    left = Lower(3)
    right = Lower(3)
    @test left == right
    @test !(left === right)
    @test left == Lower(3)
    @test left != Upper(3)
    @test left != Lower(1)
end

@testset "index hash function" begin
    @test hash(Lower(3)) == hash(Lower(3))
    @test hash(Lower(3)) != hash(Lower(1))
    @test hash(Lower(1)) != hash(Lower(3))
    @test hash(Upper(3)) != hash(Lower(3))
end

@testset "flip" begin
    @test flip(Lower(3)) == Upper(3)
    @test flip(Upper(3)) == Lower(3)
end

@testset "eliminate_indices" begin
    IdxUnion = MC.LowerOrUpperIndex

    indices = IdxUnion[Lower(9); Upper(9); Upper(3); Lower(2); Lower(1); Lower(3); Lower(2); Upper(3); Upper(9); Lower(9)]

    output = MC.eliminate_indices(indices)

    @test output == [Lower(2); Lower(1); Lower(2); Upper(3)]
    @test MC.eliminate_indices(IdxUnion[]) == IdxUnion[]
end

@testset "eliminated_indices" begin
    IdxUnion = MC.LowerOrUpperIndex

    indices = IdxUnion[Lower(9); Upper(9); Upper(3); Lower(2); Lower(1); Lower(3); Lower(2); Upper(3); Upper(9); Lower(9)]

    output = MC.eliminated_indices(indices)

    @test output == [9; 3]
    @test MC.eliminated_indices(IdxUnion[]) == IdxUnion[]
end

@testset "get_free_indices 1" begin
    xt = Sym("x", [Lower(1)], Zero()) # row vector
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    op1 = MC.BinaryOperation(*, xt, A)
    op2 = MC.BinaryOperation(*, A, xt)

    @test MC.get_free_indices(op1) == [Lower(2)]
    @test MC.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices 2" begin
    x = Sym("x", [Upper(1)], Zero())
    δ = KrD([Lower(1); Lower(2)])

    op1 = MC.BinaryOperation(*, x, δ)
    op2 = MC.BinaryOperation(*, δ, x)

    @test MC.get_free_indices(op1) == [Lower(2)]
    @test MC.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices 3" begin
    x = Sym("x", [Upper(1)], Zero())
    δ = KrD([Lower(1); Lower(1)])

    op1 = MC.BinaryOperation(*, x, δ)
    op2 = MC.BinaryOperation(*, δ, x)

    @test MC.get_free_indices(op1) == [Lower(1)]
    @test MC.get_free_indices(op2) == [Lower(1)]
end

@testset "get_free_indices 4" begin
    x = Sym("x", [], Zero())
    δ = KrD([Lower(1); Lower(1)])

    op1 = MC.BinaryOperation(*, x, δ)
    op2 = MC.BinaryOperation(*, δ, x)

    @test MC.get_free_indices(op1) == [Lower(1); Lower(1)]
    @test MC.get_free_indices(op2) == [Lower(1); Lower(1)]
end

@testset "can_contract 1" begin
    x = Sym("x", [Upper(1)], Zero())
    y = Sym("y", [Lower(1)], Zero())

    @test MC.can_contract(x, y)
    @test MC.can_contract(y, x)
end

# TODO: can_contract == true for
# BinaryOperation(*, UnaryOperation(KrD(Union{Lower, Upper}[Lower(1), Lower(1)]), UnaryOperation(KrD(Union{Lower, Upper}[Lower(3), Lower(1)]), Sym("x", Union{Lower, Upper}[Upper(3)], KrD(Union{Lower, Upper}[Upper(2), Lower(3)])))), Sym("A", Union{Lower, Upper}[Upper(1), Lower(2)], Zero()))

@testset "can_contract 2" begin
    x = Sym("x", [Lower(1)], Zero())
    y = Sym("y", [Lower(1)], Zero())

    @test !MC.can_contract(x, y)
    @test !MC.can_contract(y, x)
end

@testset "can_contract 3" begin
    x = Sym("x", [Upper(2)], Zero())
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    @test MC.can_contract(A, x)
    @test MC.can_contract(x, A)
end

@testset "can_contract 4" begin
    x = Sym("x", [Lower(3)], Zero())
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    @test !MC.can_contract(A, x)
    @test !MC.can_contract(x, A)
end


@testset "create BinaryOperation with matching indices" begin
    x = Sym("x", [Upper(2)], Zero())
    y = Sym("y", [Lower(1)], Zero())
    z = Sym("z", [], Zero())
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    @test typeof(A * x) == MC.BinaryOperation
    @test typeof(y * A) == MC.BinaryOperation
    @test typeof(z * A) == MC.BinaryOperation
    @test typeof(A * z) == MC.BinaryOperation
end

@testset "match_index column vector" begin
    x = Sym("x", [Upper(3)], Zero())

    @test MC.match_index(x, Upper(3)) == x

    expected_shift = KrD([Lower(3); Upper(1)])
    @test MC.match_index(x, Lower(1)) == MC.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Lower(3); Upper(2)])
    @test MC.match_index(x, Lower(2)) == MC.UnaryOperation(expected_shift, x)

    # match_index shall not transpose
    @test_throws DomainError MC.match_index(x, Upper(1))
end

@testset "match_index row vector" begin
    x = Sym("x", [Lower(3)], Zero())

    @test MC.match_index(x, Lower(3)) == x

    expected_shift = KrD([Upper(3); Lower(1)])
    @test MC.match_index(x, Upper(1)) == MC.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(3); Lower(2)])
    @test MC.match_index(x, Upper(2)) == MC.UnaryOperation(expected_shift, x)

    # match_index shall not transpose
    @test_throws DomainError MC.match_index(x, Lower(1))
end

@testset "match_index matrix" begin
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    @test MC.match_index(A, Lower(2)) == A

    expected_shift = KrD([Upper(2); Lower(3)])
    @test MC.match_index(A, Upper(3)) == MC.UnaryOperation(expected_shift, A)

    # match_index shall not transpose
    @test_throws DomainError MC.match_index(A, Lower(3))
end

@testset "transpose vector" begin
    x = Sym("x", [Upper(1)], Zero())
    y = Sym("y", [Lower(1)], Zero())

    expected_shift = KrD([Lower(1); Lower(1)])
    @test x' == MC.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(1); Upper(1)])
    @test y' == MC.UnaryOperation(expected_shift, y)
end

@testset "combined match_index and transpose vector" begin
    x = Sym("x", [Upper(2)], Zero())

    updated_transpose = MC.match_index(x', Lower(1))

    expected_first_shift = KrD([Lower(2); Upper(1)])
    expected_second_shift = KrD([Lower(1); Lower(1)])
    @test typeof(updated_transpose) == MC.UnaryOperation
    @test updated_transpose.op == expected_second_shift
    @test typeof(updated_transpose.arg) == MC.UnaryOperation
    @test updated_transpose.arg.arg == x
    @test updated_transpose.arg.op == expected_first_shift
end

# TODO: This is actually vector * diag(matrix) (or trace(matrix))
# Should implement
@testset "create BinaryOperation with ambigous indices fails" begin
    x = Sym("x", [Upper(2)], Zero())
    A = Sym("A", [Upper(2); Lower(2)], Zero())

    @test_throws DomainError A * x
    @test_throws DomainError x' * A
end

# TODO: Not implemented
# @testset "transpose matrix" begin
#     A = Sym("A", [Upper(1); Lower(2)], [])

#     expected_first_shift = KrD([Upper(2); Upper(2)])
#     expected_second_shift = KrD([Lower(1); Lower(1)])
#     A_transpose = A'
#     @test typeof(A_transpose) == MC.UnaryOperation
#     @test typeof(A_transpose.op) == expected_second_shift
#     @test typeof(A_transpose.arg) == MC.UnaryOperation
#     @test typeof(A_transpose.arg.op) == expected_first_shift
#     @test typeof(A_transpose.arg.arg) == A
# end

@testset "create BinaryOperation with non-matching indices matrix-vector" begin
    x = Sym("x", [Upper(3)], Zero())
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    op1 = A * x

    @test typeof(op1) == MC.BinaryOperation
    @test MC.can_contract(op1.arg1, op1.arg2)
    @test op1.op == *
    @test typeof(op1.arg1) == MC.UnaryOperation
    @test op1.arg2 == x

    # TODO: Need to transpose the target index when passing through a UnaryOperation(transpose, _)
    op2 = x' * A

    @test typeof(op2) == MC.BinaryOperation
    @test MC.can_contract(op2.arg1, op2.arg2)
    @test op2.op == *
    @test typeof(op2.arg1) == MC.UnaryOperation
    @test typeof(op2.arg1.arg) == MC.UnaryOperation
    @test op2.arg1.arg.arg == x
    @test op2.arg1.arg.op == KrD([Lower(3); Upper(1)])
    @test op2.arg1.op == KrD([Lower(1); Lower(1)])
    @test op2.arg2 == A
end

@testset "create BinaryOperation with non-compatible matrix-vector fails" begin
    x = Sym("x", [Upper(3)], Zero())
    A = Sym("A", [Upper(1); Lower(2)], Zero())

    @test_throws DomainError x * A
end

@testset "create BinaryOperation with non-matching indices vector-vector" begin
    x = Sym("x", [Upper(2)], Zero())
    y = Sym("y", [Upper(1)], Zero())

    op1 = x' * y

    @test typeof(op1) == MC.BinaryOperation
    @test MC.can_contract(op1.arg1, op1.arg2)
    @test op1.op == *
    @test typeof(op1.arg1) == MC.UnaryOperation
    @test op1.arg2 == y

    op2 = y' * x

    @test typeof(op2) == MC.BinaryOperation
    @test MC.can_contract(op2.arg1, op2.arg2)
    @test op2.op == *
    @test typeof(op2.arg1) == MC.UnaryOperation
    @test op2.arg2 == x
end

@testset "create BinaryOperation with non-matching indices scalar-matrix" begin
    A = Sym("A", [Upper(1); Lower(2)], Zero())
    z = Sym("z", [], Zero())

    op1 = A * z
    op2 = z * A

    @test typeof(op1) == MC.BinaryOperation
    @test MC.can_contract(op1.arg1, op1.arg2)
    @test op1.op == *
    @test op1.arg1 == A
    @test op1.arg2 == z

    @test typeof(op2) == MC.BinaryOperation
    @test MC.can_contract(op2.arg1, op2.arg2)
    @test op2.op == *
    @test op2.arg1 == z
    @test op2.arg2 == A
end

@testset "create BinaryOperation with non-matching indices scalar-vector" begin
    x = Sym("x", [Upper(3)], Zero())
    z = Sym("z", [], Zero())

    op1 = z * x
    op2 = x * z

    @test typeof(op1) == MC.BinaryOperation
    @test MC.can_contract(op1.arg1, op1.arg2)
    @test op1.op == *
    @test op1.arg1 == z
    @test op1.arg2 == x

    @test typeof(op2) == MC.BinaryOperation
    @test MC.can_contract(op2.arg1, op2.arg2)
    @test op2.op == *
    @test op2.arg1 == x
    @test op2.arg2 == z
end

@testset "evaluate Sym" begin
    A = Sym("A", [Upper(1); Lower(2)], Zero())
    x = Sym("x", [Upper(3)], Zero())
    z = Sym("z", [], Zero())

    @test evaluate(A) == A
    @test evaluate(x) == x
    @test evaluate(z) == z
end

@testset "evaluate KrD simple" begin
    A = Sym("A", [Upper(1); Lower(2)], Zero())
    x = Sym("x", [Upper(1)], Zero())
    z = Sym("z", [], Zero())

    d1 = KrD([Lower(1); Upper(3)])
    d2 = KrD([Upper(2); Lower(3)])

    @test evaluate(MC.UnaryOperation(d1, A)) == Sym("A", [Upper(3); Lower(2)], Zero())
    @test evaluate(MC.UnaryOperation(d1, x)) == Sym("x", [Upper(3)], Zero())
    # @test evaluate(MC.UnaryOperation(d1, z)) == Sym("z", [], Zero()) # Not implemented

    @test evaluate(MC.UnaryOperation(d2, A)) == Sym("A", [Upper(1); Lower(3)], Zero())
end

@testset "evaluate transpose simple" begin
    A = Sym("A", [Upper(1); Lower(2)], Zero())
    x = Sym("x", [Upper(1)], Zero())
    z = Sym("z", [], Zero())

    # @test evaluate(A') == Sym("A", [Upper(2); Lower(1)], Zero()) # Not implemented
    @test evaluate(x') == Sym("x", [Lower(1)], Zero())
    # @test evaluate(z') == Sym("z", [], Zero()) # Not implemented
end

# TODO: Not implemented
# @testset "evaluate KrD 2" begin
#     B = Sym("B", [Upper(2); Lower(1)], Zero())
#     x = Sym("x", [Upper(2)], Zero())
#     z = Sym("z", [], Zero())

#     d1 = KrD([Lower(1); Upper(3)])

#     @test evaluate(MC.UnaryOperation(d1, B)) == MC.UnaryOperation(d1, B)
#     @test evaluate(MC.UnaryOperation(d1, x)) == MC.UnaryOperation(d1, x)
# end