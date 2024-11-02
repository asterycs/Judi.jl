using MatrixDiff
using Test

MD = MatrixDiff

@testset "Sym constructor throws on invalid input" begin
    @test_throws DomainError Sym("A", [Upper(2); Lower(2)])
    @test_throws DomainError Sym("A", [Lower(2); Lower(2)])

    A = Sym("A", [Upper(1); Lower(2)])
    B = Sym("B", [Lower(1); Lower(2)])
    x = Sym("x", [Upper(1)])
    y = Sym("y", [Lower(1)])
    z = Sym("z", [])
end

@testset "index equality operator" begin
    left = Lower(3)
    right = Lower(3)
    @test left == right
    @test !(left === right)
    @test left == Lower(3)
    @test left != Upper(3)
    @test left != Lower(1)
end

@testset "KrD equality operator" begin
    left = KrD([Upper(1); Lower(2)])
    @test KrD([Upper(1); Lower(2)]) == KrD([Upper(1); Lower(2)])
    @test !(KrD([Upper(1); Lower(2)]) === KrD([Upper(1); Lower(2)]))
    @test left == KrD([Upper(1); Lower(2)])
    @test left != KrD([Upper(1); Upper(2)])
    @test left != KrD([Lower(1); Lower(2)])
    @test left != KrD([Upper(2); Lower(2)])
    @test left != KrD([Upper(1); Lower(3)])
    @test left != KrD([Upper(1)])
    @test left != KrD([Upper(1); Lower(2); Lower(3)])
end


@testset "UnaryOperation equality operator" begin
    a = KrD([Upper(1); Upper(1)])
    b = Sym("b", [Lower(1)])

    left = MD.UnaryOperation(a, b)

    @test MD.UnaryOperation(a, b) == MD.UnaryOperation(a, b)
    @test !(MD.UnaryOperation(a, b) === MD.UnaryOperation(a, b))
    @test left == MD.UnaryOperation(a, b)
    @test left != MD.UnaryOperation(b, a)
end

@testset "BinaryOperation equality operator" begin
    a = Sym("a", [Upper(1)])
    b = Sym("b", [Lower(1)])

    left = MD.BinaryOperation(*, a, b)

    @test MD.BinaryOperation(*, a, b) == MD.BinaryOperation(*, a, b)
    @test !(MD.BinaryOperation(*, a, b) === MD.BinaryOperation(*, a, b))
    @test left == MD.BinaryOperation(*, a, b)
    @test left == MD.BinaryOperation(*, b, a)
    @test left != MD.BinaryOperation(+, a, b)
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

@testset "eliminate_indices removes correct indices" begin
    IdxUnion = MD.LowerOrUpperIndex

    indices = IdxUnion[Lower(9); Upper(9); Upper(3); Lower(2); Lower(1); Lower(3); Lower(2); Upper(3); Upper(9); Lower(9)]

    output = MD.eliminate_indices(indices)

    @test output == [Lower(2); Lower(1); Lower(2); Upper(3)]
    @test MD.eliminate_indices(IdxUnion[]) == IdxUnion[]
end

@testset "eliminated_indices retains correct indices" begin
    IdxUnion = MD.LowerOrUpperIndex

    indices = IdxUnion[Lower(9); Upper(9); Upper(3); Lower(2); Lower(1); Lower(3); Lower(2); Upper(3); Upper(9); Lower(9)]

    output = MD.eliminated_indices(indices)

    @test output == IdxUnion[Lower(9); Upper(9); Upper(3); Lower(3); Upper(9); Lower(9)]
    @test MD.eliminated_indices(IdxUnion[]) == IdxUnion[]
end

@testset "get_free_indices with Sym-Sym and one matching pair" begin
    xt = Sym("x", [Lower(1)]) # row vector
    A = Sym("A", [Upper(1); Lower(2)])

    op1 = MD.BinaryOperation(*, xt, A)
    op2 = MD.BinaryOperation(*, A, xt)

    @test MD.get_free_indices(op1) == [Lower(2)]
    @test MD.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Sym-KrD and one matching pair" begin
    x = Sym("x", [Upper(1)])
    δ = KrD([Lower(1); Lower(2)])

    op1 = MD.BinaryOperation(*, x, δ)
    op2 = MD.BinaryOperation(*, δ, x)

    @test MD.get_free_indices(op1) == [Lower(2)]
    @test MD.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Sym-KrD and two matching pairs" begin
    x = Sym("x", [Upper(1)])
    δ = KrD([Lower(1); Lower(1)])

    op1 = MD.BinaryOperation(*, x, δ)
    op2 = MD.BinaryOperation(*, δ, x)

    @test MD.get_free_indices(op1) == [Lower(1)]
    @test MD.get_free_indices(op2) == [Lower(1)]
end

@testset "get_free_indices with scalar Sym-KrD" begin
    x = Sym("x", [])
    δ = KrD([Lower(1); Lower(1)])

    op1 = MD.BinaryOperation(*, x, δ)
    op2 = MD.BinaryOperation(*, δ, x)

    @test MD.get_free_indices(op1) == [Lower(1); Lower(1)]
    @test MD.get_free_indices(op2) == [Lower(1); Lower(1)]
end

@testset "get_free_indices with Sym-Sym and no matching pairs" begin
    x = Sym("x", [Upper(1)])
    A = Sym("A", [Upper(1); Lower(2)])

    op1 = MD.BinaryOperation(*, x, A)
    op2 = MD.BinaryOperation(*, A, x)

    @test MD.get_free_indices(op1) == [Upper(1); Upper(1); Lower(2)]
    @test MD.get_free_indices(op2) == [Upper(1); Lower(2); Upper(1)]
end

@testset "is_contraction_unambigous vector-vector with matching pair" begin
    x = Sym("x", [Upper(1)])
    y = Sym("y", [Lower(1)])

    @test MD.is_contraction_unambigous(x, y)
    @test MD.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous vector-vector with non-matching pair" begin
    x = Sym("x", [Lower(1)])
    y = Sym("y", [Lower(1)])

    @test !MD.is_contraction_unambigous(x, y)
    @test !MD.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous matrix-vector with matching pair" begin
    x = Sym("x", [Upper(2)])
    A = Sym("A", [Upper(1); Lower(2)])

    @test MD.is_contraction_unambigous(A, x)
    @test MD.is_contraction_unambigous(x, A)
end

@testset "is_contraction_unambigous matrix-vector with non-matching pair" begin
    x = Sym("x", [Lower(3)])
    A = Sym("A", [Upper(1); Lower(2)])

    @test MD.is_contraction_unambigous(A, x)
    @test MD.is_contraction_unambigous(x, A)
end

@testset "create BinaryOperation with impossible input fails" begin
    # This input cannot be written with standard matrix notation.
    # We catch it in *() although the check should arguably be somewhere else.
    A = Sym("A", [Upper(1); Lower(2)])
    x = Sym("x", [Lower(1)])
    y = Sym("y", [Upper(1)])

    @test_throws DomainError A * x
    @test_throws DomainError A * y
end

@testset "create BinaryOperation with matching indices" begin
    x = Sym("x", [Upper(2)])
    y = Sym("y", [Lower(1)])
    z = Sym("z", [])
    A = Sym("A", [Upper(1); Lower(2)])

    @test typeof(A * x) == MD.BinaryOperation
    @test typeof(y * A) == MD.BinaryOperation
    @test typeof(z * A) == MD.BinaryOperation
    @test typeof(A * z) == MD.BinaryOperation
end

@testset "update_index column vector" begin
    x = Sym("x", [Upper(3)])

    @test MD.update_index(x, Upper(3), Upper(3)) == x

    expected_shift = KrD([Lower(3); Upper(1)])
    @test MD.update_index(x, Upper(3), Upper(1)) == MD.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Lower(3); Upper(2)])
    @test MD.update_index(x, Upper(3), Upper(2)) == MD.UnaryOperation(expected_shift, x)

    # update_index shall not transpose
    @test_throws DomainError MD.update_index(x, Upper(3), Lower(1))
end

@testset "update_index row vector" begin
    x = Sym("x", [Lower(3)])

    @test MD.update_index(x, Lower(3), Lower(3)) == x

    expected_shift = KrD([Upper(3); Lower(1)])
    @test MD.update_index(x, Lower(3), Lower(1)) == MD.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(3); Lower(2)])
    @test MD.update_index(x, Lower(3), Lower(2)) == MD.UnaryOperation(expected_shift, x)

    # update_index shall not transpose
    @test_throws DomainError MD.update_index(x, Lower(3), Upper(1))
end

@testset "update_index matrix" begin
    A = Sym("A", [Upper(1); Lower(2)])

    @test MD.update_index(A, Lower(2), Lower(2)) == A

    expected_shift = KrD([Upper(2); Lower(3)])
    @test MD.update_index(A, Lower(2), Lower(3)) == MD.UnaryOperation(expected_shift, A)

    # update_index shall not transpose
    @test_throws DomainError MD.update_index(A, Lower(2), Upper(3))
end

@testset "transpose vector" begin
    x = Sym("x", [Upper(1)])
    y = Sym("y", [Lower(1)])

    expected_shift = KrD([Lower(1); Lower(1)])
    @test x' == MD.UnaryOperation(expected_shift, x)

    expected_shift = KrD([Upper(1); Upper(1)])
    @test y' == MD.UnaryOperation(expected_shift, y)
end

@testset "combined update_index and transpose vector" begin
    x = Sym("x", [Upper(2)])

    updated_transpose = MD.update_index(x', Lower(2), Lower(1))

    expected_first_shift = KrD([Lower(2); Lower(2)])
    expected_second_shift = KrD([Upper(2); Lower(1)])
    @test typeof(updated_transpose) == MD.UnaryOperation
    @test updated_transpose.op == expected_second_shift
    @test typeof(updated_transpose.arg) == MD.UnaryOperation
    @test updated_transpose.arg.arg == x
    @test updated_transpose.arg.op == expected_first_shift
end

# TODO: Not implemented
# @testset "transpose matrix" begin
#     A = Sym("A", [Upper(1); Lower(2)], [])

#     expected_first_shift = KrD([Upper(2); Upper(2)])
#     expected_second_shift = KrD([Lower(1); Lower(1)])
#     A_transpose = A'
#     @test typeof(A_transpose) == MD.UnaryOperation
#     @test typeof(A_transpose.op) == expected_second_shift
#     @test typeof(A_transpose.arg) == MD.UnaryOperation
#     @test typeof(A_transpose.arg.op) == expected_first_shift
#     @test typeof(A_transpose.arg.arg) == A
# end

@testset "can_contract" begin
    A = Sym("A", [Upper(1); Lower(2)])
    x = Sym("x", [Upper(2)])
    y = Sym("y", [Upper(3)])
    z = Sym("z", [Lower(1)])
    d = KrD([Lower(1); Lower(1)])

    @test MD.can_contract(A, x)
    @test MD.can_contract(x, A)
    @test !MD.can_contract(A, y)
    @test !MD.can_contract(y, A)
    @test MD.can_contract(A, d)
    @test MD.can_contract(d, A)
    @test MD.can_contract(A, z)
    @test MD.can_contract(z, A)
end

@testset "create BinaryOperation with non-matching indices matrix-vector" begin
    x = Sym("x", [Upper(3)])
    A = Sym("A", [Upper(1); Lower(2)])

    op1 = A * x

    @test typeof(op1) == MD.BinaryOperation
    @test MD.can_contract(op1.arg1, op1.arg2)
    @test op1.op == *
    @test typeof(op1.arg1) == MD.UnaryOperation
    @test op1.arg2 == x

    op2 = x' * A

    @test typeof(op2) == MD.BinaryOperation
    @test MD.can_contract(op2.arg1, op2.arg2)
    @test op2.op == *
    @test typeof(op2.arg1) == MD.UnaryOperation
    @test typeof(op2.arg1.arg) == MD.UnaryOperation
    @test op2.arg1.arg.arg == x
    @test op2.arg1.arg.op == KrD([Lower(3); Lower(3)])
    @test op2.arg1.op == KrD([Upper(3); Lower(1)])
    @test op2.arg2 == A
end

@testset "create BinaryOperation with non-compatible matrix-vector fails" begin
    x = Sym("x", [Upper(3)])
    A = Sym("A", [Upper(1); Lower(2)])

    @test_throws DomainError x * A
end

@testset "create BinaryOperation with non-matching indices vector-vector" begin
    x = Sym("x", [Upper(2)])
    y = Sym("y", [Upper(1)])

    op1 = x' * y

    @test typeof(op1) == MD.BinaryOperation
    @test MD.can_contract(op1.arg1, op1.arg2)
    @test op1.op == *
    @test typeof(op1.arg1) == MD.UnaryOperation
    @test op1.arg2 == y

    op2 = y' * x

    @test typeof(op2) == MD.BinaryOperation
    @test MD.can_contract(op2.arg1, op2.arg2)
    @test op2.op == *
    @test typeof(op2.arg1) == MD.UnaryOperation
    @test op2.arg2 == x
end

# TODO: Scalars not implemented
# @testset "create BinaryOperation with non-matching indices scalar-matrix" begin
#     A = Sym("A", [Upper(1); Lower(2)])
#     z = Sym("z", [])

#     op1 = A * z
#     op2 = z * A

#     @test typeof(op1) == MD.BinaryOperation
#     @test MD.can_contract(op1.arg1, op1.arg2)
#     @test op1.op == *
#     @test op1.arg1 == A
#     @test op1.arg2 == z

#     @test typeof(op2) == MD.BinaryOperation
#     @test MD.can_contract(op2.arg1, op2.arg2)
#     @test op2.op == *
#     @test op2.arg1 == z
#     @test op2.arg2 == A
# end

# @testset "create BinaryOperation with non-matching indices scalar-vector" begin
#     x = Sym("x", [Upper(3)])
#     z = Sym("z", [])

#     op1 = z * x
#     op2 = x * z

#     @test typeof(op1) == MD.BinaryOperation
#     @test MD.can_contract(op1.arg1, op1.arg2)
#     @test op1.op == *
#     @test op1.arg1 == z
#     @test op1.arg2 == x

#     @test typeof(op2) == MD.BinaryOperation
#     @test MD.can_contract(op2.arg1, op2.arg2)
#     @test op2.op == *
#     @test op2.arg1 == x
#     @test op2.arg2 == z
# end
