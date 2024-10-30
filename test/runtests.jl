using MatrixCalculus
using Test

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
    IdxUnion = MatrixCalculus.LowerOrUpperIndex

    indices = IdxUnion[Lower(9); Upper(9); Upper(3); Lower(2); Lower(1); Lower(3); Lower(2); Upper(3); Upper(9); Lower(9)]

    output = eliminate_indices(indices)

    @test output == [Lower(2); Lower(1); Lower(2); Upper(3)]
    @test eliminate_indices(IdxUnion[]) == IdxUnion[]
end

@testset "eliminated_indices" begin
    IdxUnion = MatrixCalculus.LowerOrUpperIndex

    indices = IdxUnion[Lower(9); Upper(9); Upper(3); Lower(2); Lower(1); Lower(3); Lower(2); Upper(3); Upper(9); Lower(9)]

    output = eliminated_indices(indices)

    @test output == [9; 3]
    @test eliminated_indices(IdxUnion[]) == IdxUnion[]
end

@testset "get_free_indices 1" begin
    xt = Sym("x", [Lower(1)], []) # row vector
    A = Sym("A", [Upper(1); Lower(2)], [])

    op1 = MatrixCalculus.BinaryOperation(*, xt, A)
    op2 = MatrixCalculus.BinaryOperation(*, A, xt)

    @test MatrixCalculus.get_free_indices(op1) == [Lower(2)]
    @test MatrixCalculus.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices 2" begin
    x = Sym("x", [Upper(1)], [])
    δ = KrD([Lower(1); Lower(2)])

    op1 = MatrixCalculus.BinaryOperation(*, x, δ)
    op2 = MatrixCalculus.BinaryOperation(*, δ, x)

    @test MatrixCalculus.get_free_indices(op1) == [Lower(2)]
    @test MatrixCalculus.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices 3" begin
    x = Sym("x", [Upper(1)], [])
    δ = KrD([Lower(1); Lower(1)])

    op1 = MatrixCalculus.BinaryOperation(*, x, δ)
    op2 = MatrixCalculus.BinaryOperation(*, δ, x)

    @test MatrixCalculus.get_free_indices(op1) == [Lower(1)]
    @test MatrixCalculus.get_free_indices(op2) == [Lower(1)]
end

@testset "get_free_indices 4" begin
    x = Sym("x", [], [])
    δ = KrD([Lower(1); Lower(1)])

    op1 = MatrixCalculus.BinaryOperation(*, x, δ)
    op2 = MatrixCalculus.BinaryOperation(*, δ, x)

    @test MatrixCalculus.get_free_indices(op1) == [Lower(1); Lower(1)]
    @test MatrixCalculus.get_free_indices(op2) == [Lower(1); Lower(1)]
end

@testset "can_contract 1" begin
    x = Sym("x", [Upper(1)], [])
    y = Sym("y", [Lower(1)], [])

    @test MatrixCalculus.can_contract(x, y)
    @test MatrixCalculus.can_contract(y, x)
end

@testset "can_contract 2" begin
    x = Sym("x", [Lower(1)], [])
    y = Sym("y", [Lower(1)], [])

    @test !MatrixCalculus.can_contract(x, y)
    @test !MatrixCalculus.can_contract(y, x)
end

@testset "can_contract 3" begin
    x = Sym("x", [Upper(2)], [])
    A = Sym("A", [Upper(1); Lower(2)], [])

    @test MatrixCalculus.can_contract(A, x)
    @test MatrixCalculus.can_contract(x, A)
end

@testset "can_contract 4" begin
    x = Sym("x", [Lower(3)], [])
    A = Sym("A", [Upper(1); Lower(2)], [])

    @test !MatrixCalculus.can_contract(A, x)
    @test !MatrixCalculus.can_contract(x, A)
end


@testset "create BinaryOperation with matching indices" begin
    x = Sym("x", [Upper(2)], [])
    y = Sym("y", [Lower(1)], [])
    z = Sym("z", [], [])
    A = Sym("A", [Upper(1); Lower(2)], [])

    @test typeof(A * x) == MatrixCalculus.BinaryOperation
    @test typeof(y * A) == MatrixCalculus.BinaryOperation
    @test typeof(z * A) == MatrixCalculus.BinaryOperation
    @test typeof(A * z) == MatrixCalculus.BinaryOperation
end

@testset "update_index column vector" begin
    x = Sym("x", [Upper(3)], [])

    @test MatrixCalculus.update_index(x, Upper(3)) == x

    expected_shift = KrD([Lower(3); Upper(1)])
    @test MatrixCalculus.update_index(x, Upper(1)) == MatrixCalculus.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Lower(3); Lower(3)])
    @test MatrixCalculus.update_index(x, Lower(3)) == MatrixCalculus.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Lower(3); Lower(2)])
    @test MatrixCalculus.update_index(x, Lower(2)) == MatrixCalculus.UnaryOperation(expected_shift, x)
end

@testset "update_index row vector" begin
    x = Sym("x", [Lower(3)], [])

    @test MatrixCalculus.update_index(x, Lower(3)) == x

    expected_shift = KrD([Upper(3); Lower(1)])
    @test MatrixCalculus.update_index(x, Lower(1)) == MatrixCalculus.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(3); Upper(3)])
    @test MatrixCalculus.update_index(x, Upper(3)) == MatrixCalculus.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(3); Upper(2)])
    @test MatrixCalculus.update_index(x, Upper(2)) == MatrixCalculus.UnaryOperation(expected_shift, x)
end

@testset "update_index matrix" begin
    A = Sym("A", [Upper(1); Lower(2)], [])

    @test MatrixCalculus.update_index(A, Lower(2)) == A

    expected_shift = KrD([Upper(2); Upper(3)])
    @test MatrixCalculus.update_index(A, Upper(3)) == MatrixCalculus.UnaryOperation(expected_shift, A)
end

@testset "adjoint vector" begin
    x = Sym("x", [Upper(1)], [])
    y = Sym("y", [Lower(1)], [])

    expected_shift = KrD([Lower(1); Lower(1)])
    @test x' == MatrixCalculus.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(1); Upper(1)])
    @test y' == MatrixCalculus.UnaryOperation(expected_shift, y)
end

@testset "combined update_index and adjoint vector" begin
    x = Sym("x", [Upper(2)], [])

    updated_adjoint = MatrixCalculus.update_index(x', Upper(1))

    expected_first_shift = KrD([Lower(2); Upper(1)])
    expected_second_shift = KrD([Lower(1); Lower(1)])
    @test typeof(updated_adjoint) == MatrixCalculus.UnaryOperation
    @test updated_adjoint.op == expected_second_shift
    @test typeof(updated_adjoint.op.arg) == MatrixCalculus.UnaryOperation
    @test updated_adjoint.arg.op == expected_first_shift
end

# TODO: Not implemented
# @testset "adjoint matrix" begin
#     A = Sym("A", [Upper(1); Lower(2)], [])

#     expected_first_shift = KrD([Upper(2); Upper(2)])
#     expected_second_shift = KrD([Lower(1); Lower(1)])
#     A_adjoint = A'
#     @test typeof(A_adjoint) == MatrixCalculus.UnaryOperation
#     @test typeof(A_adjoint.op) == expected_second_shift
#     @test typeof(A_adjoint.arg) == MatrixCalculus.UnaryOperation
#     @test typeof(A_adjoint.arg.op) == expected_first_shift
#     @test typeof(A_adjoint.arg.arg) == A
# end

# @testset "create BinaryOperation with non-matching indices" begin
#     x = Sym("x", [Upper(3)], [])
#     y = Sym("y", [Lower(1)], [])
#     z = Sym("z", [], [])
#     A = Sym("A", [Upper(1); Lower(2)], [])

#     op1 = A * x

#     @test typeof(op1) == MatrixCalculus.BinaryOperation
#     @test MatrixCalculus.can_contract(op1.arg1, op1.arg2)
# end