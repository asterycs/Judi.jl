using MatrixDiff
using Test

MD = MatrixDiff

@testset "Sym constructor creates a valid Sym" begin
    A = Sym("A", Upper(), Lower())
    @test A.id == "A"
    @test A.indices == [Upper(); Lower()]

    B = Sym("B", Lower(), Lower())
    @test B.id == "B"
    @test B.indices == [Lower(); Lower()]

    x = Sym("x", Upper())
    @test x.id == "x"
    @test x.indices == [Upper()]

    y = Sym("y", Lower())
    @test y.id == "y"
    @test y.indices == [Lower()]

    z = Sym("z")
    @test z.id == "z"
    @test isempty(z.indices)
end

@testset "index equality operator" begin
    left = Lower()
    right = Lower()
    @test left == right
    @test left == Lower()
    @test left != Upper()
end

@testset "KrD constructor throws on invalid input" begin
    @test_throws DomainError KrD(Upper())
    @test_throws DomainError KrD()
end

@testset "KrD constructor creates a valid KrD" begin
    d = KrD(Upper(), Upper())
    @test d.indices == [Upper(); Upper()]

    d2 = KrD(Upper(), Lower(), Upper())
    @test d2.indices == [Upper(); Lower(); Upper()]
end

@testset "KrD equality operator" begin
    left = KrD(Upper(), Lower())
    @test KrD(Upper(), Lower()) == KrD(Upper(), Lower())
    @test !(KrD(Upper(), Lower()) === KrD(Upper(), Lower()))
    @test left == KrD(Upper(), Lower())
    @test left != KrD(Upper(), Upper())
    @test left != KrD(Lower(), Lower())
    @test left != KrD(Upper(), Lower(), Lower())
end

@testset "Zero equality operator" begin
    left = Zero(Upper(), Lower())
    @test Zero(Upper(), Lower()) == Zero(Upper(), Lower())
    @test !(Zero(Upper(), Lower()) === Zero(Upper(), Lower()))
    @test left == Zero(Upper(), Lower())
    @test left != Zero(Upper(), Upper())
    @test left != Zero(Lower(), Lower())
    @test left != Zero(Upper())
    @test left != Zero(Upper(), Lower(), Lower())
    @test left != Zero()
end

@testset "UnaryOperation equality operator" begin
    a = KrD(Upper(), Upper())
    b = Sym("b", Lower())

    left = MD.UnaryOperation(a, b)

    @test MD.UnaryOperation(a, b) == MD.UnaryOperation(a, b)
    @test left == MD.UnaryOperation(a, b)
    @test left != MD.UnaryOperation(b, a)
end

@testset "BinaryOperation equality operator" begin
    a = Sym("a", Upper())
    b = Sym("b", Lower())

    left = MD.BinaryOperation{*}(a, b, [(1, 1)])

    @test MD.BinaryOperation{*}(a, b, [(1, 1)]) == MD.BinaryOperation{*}(a, b, [(1, 1)])
    @test left == MD.BinaryOperation{*}(a, b, [(1, 1)])
    @test left == MD.BinaryOperation{*}(b, a, [(1, 1)])
    @test left != MD.BinaryOperation{+}(a, b, [(1, 2)])
end

@testset "index hash function" begin
    @test hash(Lower()) == hash(Lower())
    @test hash(Upper()) != hash(Lower())
end

@testset "flip" begin
    @test flip(Lower()) == Upper()
    @test flip(Upper()) == Lower()
end

@testset "eliminate_indices removes correct indices" begin
    IdxUnion = MD.LowerOrUpperIndex

    arg1 = IdxUnion[Upper(); Lower(); Lower()]
    arg2 = IdxUnion[Upper(); Lower(); Lower()]
    contractions = [(2, 1)]

    output = MD.eliminate_indices(arg1, arg2, contractions)

    @test output == [Upper(); Lower(); Lower(); Lower()]
end

@testset "eliminated_indices retains correct indices" begin
    IdxUnion = MD.LowerOrUpperIndex

    arg1 = IdxUnion[Upper(); Lower(); Lower()]
    arg2 = IdxUnion[Upper(); Lower(); Lower()]
    contractions = [(2, 1)]

    output = MD.eliminated_indices(arg1, arg2, contractions)

    @test output == [(2, 1)]
end

@testset "get_free_indices with Sym-Sym" begin
    xt = Sym("x", Lower()) # row vector
    A = Sym("A", Upper(), Lower())

    op1 = MD.BinaryOperation{*}(xt, A, [(1, 1)])
    op2 = MD.BinaryOperation{*}(A, xt, [(1, 1)]) # not valid standard notation

    @test MD.get_free_indices(op1) == [Lower()]
    @test MD.get_free_indices(op2) == [Lower()]
end

@testset "get_free_indices with Sym-KrD" begin
    x = Sym("x", Upper())
    δ = KrD(Lower(), Lower())

    op1 = MD.BinaryOperation{*}(x, δ, [(1, 1)])
    op2 = MD.BinaryOperation{*}(δ, x, [(1, 1)])

    @test MD.get_free_indices(op1) == [Lower()]
    @test MD.get_free_indices(op2) == [Lower()]
end

@testset "is_valid_multiplication succeeds with valid matrix-vector input" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())
    y = Sym("y", Lower())

    # TODO: add more tests here
    @test MD.is_valid_multiplication(A, x)
    @test MD.is_valid_multiplication(y, A)
end

@testset "is_valid_multiplication fails with invalid matrix-vector input" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())
    y = Sym("y", Lower())

    @test !MD.is_valid_multiplication(x, A)
    @test !MD.is_valid_multiplication(A, y)
end

@testset "create BinaryOperation with matching indices" begin
    x = Sym("x", Upper())
    y = Sym("y", Lower())
    A = Sym("A", Upper(), Lower())

    @test typeof(A * x) == MD.BinaryOperation{*}
    @test typeof(y * A) == MD.BinaryOperation{*}
end

# @testset "create ***Operation with matrix-scalar" begin
#     x = Sym("x", Upper())
#     A = Sym("A", Upper(), Lower())
#     z = Sym("z")

#     @test typeof(A * x) == MD.BinaryOperation{*}
#     @test typeof(y * A) == MD.BinaryOperation{*}
#     @test typeof(z * A) == MD.BinaryOperation{*}
#     @test typeof(A * z) == MD.BinaryOperation{*}
# end

@testset "transpose vector" begin
    x = Sym("x", Upper())
    y = Sym("y", Lower())

    expected_shift = KrD(Lower(), Lower())
    @test x' == MD.BinaryOperation{*}(x, expected_shift, [(1, 1)])

    expected_shift = KrD(Upper(), Upper())
    @test y' == MD.BinaryOperation{*}(y, expected_shift, [(1, 1)])
end

@testset "transpose matrix" begin
    A = Sym("A", Upper(), Lower())

    expected_first_shift = KrD(Upper(), Upper())
    expected_second_shift = KrD(Lower(), Lower())
    A_transpose = A'
    @test typeof(A_transpose) == MD.BinaryOperation{*}
    @test A_transpose.arg2 == expected_second_shift
    @test typeof(A_transpose.arg1) == MD.BinaryOperation{*}
    @test A_transpose.arg1.arg2 == expected_first_shift
    @test A_transpose.arg1.arg1 == A
end

@testset "can_contract" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())
    y = Sym("y", Upper())
    z = Sym("z", Lower())
    d = KrD(Lower(), Lower())

    @test MD.can_contract(A, x, [(2, 1)])
    @test MD.can_contract(x, A, [(1, 2)])
    @test !MD.can_contract(A, y, [(1, 1)])
    @test !MD.can_contract(y, A, [(1, 1)])
    @test MD.can_contract(A, d, [(1, 1)])
    @test MD.can_contract(d, A, [(1, 1)])
    @test MD.can_contract(A, z, [(1, 1)])
    @test MD.can_contract(z, A, [(1, 1)])
end

@testset "create BinaryOperation with non-matching indices matrix-vector" begin
    x = Sym("x", Upper())
    A = Sym("A", Upper(), Lower())

    op1 = A * x

    @test typeof(op1) == MD.BinaryOperation{*}
    @test MD.can_contract(op1.arg1, op1.arg2, op1.indices)
    @test op1.arg1 == A
    @test op1.arg2 == x

    op2 = x' * A

    @test typeof(op2) == MD.BinaryOperation{*}
    @test MD.can_contract(op2.arg1, op2.arg2, op2.indices)
    @test typeof(op2.arg1) == MD.BinaryOperation{*}
    @test op2.arg1.arg1 == x
    @test op2.arg1.arg2 == KrD(Lower(), Lower())
    @test op2.arg2 == A
end

@testset "create BinaryOperation with non-compatible matrix-vector fails" begin
    x = Sym("x", Upper())
    A = Sym("A", Upper(), Lower())

    @test_throws DomainError x * A
end

@testset "create BinaryOperation with non-matching indices vector-vector" begin
    x = Sym("x", Upper())
    y = Sym("y", Upper())

    op1 = x' * y

    @test typeof(op1) == MD.BinaryOperation{*}
    @test MD.can_contract(op1.arg1, op1.arg2, op1.indices)
    @test typeof(op1.arg1) == MD.BinaryOperation{*}
    @test op1.arg2 == y

    op2 = y' * x

    @test typeof(op2) == MD.BinaryOperation{*}
    @test MD.can_contract(op2.arg1, op2.arg2, op2.indices)
    @test typeof(op2.arg1) == MD.BinaryOperation{*}
    @test op2.arg2 == x
end

# TODO: Scalars not implemented
# @testset "create BinaryOperation with non-matching indices scalar-matrix" begin
#     A = Sym("A", Upper(), Lower())
#     z = Sym("z")

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
#     x = Sym("x", Upper())
#     z = Sym("z")

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
