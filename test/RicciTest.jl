using MatrixDiff
using Test

MD = MatrixDiff

@testset "Tensor constructor throws on invalid input" begin
    @test_throws DomainError Tensor("A", Upper(2), Lower(2))
    @test_throws DomainError Tensor("A", Lower(2), Lower(2))

    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Lower(1), Lower(2))
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))
    z = Tensor("z")
end

@testset "index equality operator" begin
    left = Lower(3)
    right = Lower(3)
    @test left == right
    @test left == Lower(3)
    @test left != Upper(3)
    @test left != Lower(1)
end

@testset "KrD equality operator" begin
    left = KrD(Upper(1), Lower(2))
    @test KrD(Upper(1), Lower(2)) == KrD(Upper(1), Lower(2))
    @test !(KrD(Upper(1), Lower(2)) === KrD(Upper(1), Lower(2)))
    @test left == KrD(Upper(1), Lower(2))
    @test left != KrD(Upper(1), Upper(2))
    @test left != KrD(Lower(1), Lower(2))
    @test left != KrD(Upper(3), Lower(2))
    @test left != KrD(Upper(1), Lower(3))
    @test left != KrD(Upper(1))
    @test left != KrD(Upper(1), Lower(2), Lower(3))
end

@testset "Zero equality operator" begin
    left = Zero(Upper(1), Lower(2))
    @test Zero(Upper(1), Lower(2)) == Zero(Upper(1), Lower(2))
    @test !(Zero(Upper(1), Lower(2)) === Zero(Upper(1), Lower(2)))
    @test left == Zero(Upper(1), Lower(2))
    @test left != Zero(Upper(1), Upper(2))
    @test left != Zero(Lower(1), Lower(2))
    @test left != Zero(Upper(3), Lower(2))
    @test left != Zero(Upper(1), Lower(3))
    @test left != Zero(Upper(1))
    @test left != Zero(Upper(1), Lower(2), Lower(3))
    @test left != Zero()
end

@testset "UnaryOperation equality operator" begin
    a = KrD(Upper(1), Upper(1))
    b = Tensor("b", Lower(1))

    left = MD.UnaryOperation(a, b)

    @test MD.UnaryOperation(a, b) == MD.UnaryOperation(a, b)
    @test left == MD.UnaryOperation(a, b)
    @test left != MD.UnaryOperation(b, a)
end

@testset "BinaryOperation equality operator" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = MD.BinaryOperation{*}(a, b)

    @test MD.BinaryOperation{*}(a, b) == MD.BinaryOperation{*}(a, b)
    @test left == MD.BinaryOperation{*}(a, b)
    @test left == MD.BinaryOperation{*}(b, a)
    @test left != MD.BinaryOperation{+}(a, b)
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

@testset "get_free_indices with Tensor-Tensor and one matching pair" begin
    xt = Tensor("x", Lower(1)) # row vector
    A = Tensor("A", Upper(1), Lower(2))

    op1 = MD.BinaryOperation{*}(xt, A)
    op2 = MD.BinaryOperation{*}(A, xt)

    @test MD.get_free_indices(op1) == [Lower(2)]
    @test MD.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Tensor-KrD and one matching pair" begin
    x = Tensor("x", Upper(1))
    δ = KrD(Lower(1), Lower(2))

    op1 = MD.BinaryOperation{*}(x, δ)
    op2 = MD.BinaryOperation{*}(δ, x)

    @test MD.get_free_indices(op1) == [Lower(2)]
    @test MD.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Tensor-KrD and two matching pairs" begin
    x = Tensor("x", Upper(1))
    δ = KrD(Lower(1), Lower(1))

    op1 = MD.BinaryOperation{*}(x, δ)
    op2 = MD.BinaryOperation{*}(δ, x)

    @test MD.get_free_indices(op1) == [Lower(1)]
    @test MD.get_free_indices(op2) == [Lower(1)]
end

@testset "get_free_indices with scalar Tensor-KrD" begin
    x = Tensor("x")
    δ = KrD(Lower(1), Lower(1))

    op1 = MD.BinaryOperation{*}(x, δ)
    op2 = MD.BinaryOperation{*}(δ, x)

    @test MD.get_free_indices(op1) == [Lower(1); Lower(1)]
    @test MD.get_free_indices(op2) == [Lower(1); Lower(1)]
end

@testset "get_free_indices with Tensor-Tensor and no matching pairs" begin
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = MD.BinaryOperation{*}(x, A)
    op2 = MD.BinaryOperation{*}(A, x)

    @test MD.get_free_indices(op1) == [Upper(1); Upper(1); Lower(2)]
    @test MD.get_free_indices(op2) == [Upper(1); Lower(2); Upper(1)]
end

@testset "is_contraction_unambigous vector-vector with matching pair" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    @test MD.is_contraction_unambigous(x, y)
    @test MD.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous vector-vector with non-matching pair" begin
    x = Tensor("x", Lower(1))
    y = Tensor("y", Lower(1))

    @test !MD.is_contraction_unambigous(x, y)
    @test !MD.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous matrix-vector with matching pair" begin
    x = Tensor("x", Upper(2))
    A = Tensor("A", Upper(1), Lower(2))

    @test MD.is_contraction_unambigous(A, x)
    @test MD.is_contraction_unambigous(x, A)
end

@testset "is_contraction_unambigous matrix-vector with non-matching pair" begin
    x = Tensor("x", Lower(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test MD.is_contraction_unambigous(A, x)
    @test MD.is_contraction_unambigous(x, A)
end

@testset "is_valid_matrix_multiplication fails with invalid input" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(2))

    @test !MD.is_valid_matrix_multiplication(A, x)
    @test !MD.is_valid_matrix_multiplication(x, A)
    @test !MD.is_valid_matrix_multiplication(A, y)
    @test !MD.is_valid_matrix_multiplication(y, A)
end

@testset "is_valid_matrix_multiplication succeeds with valid input" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z", Lower(2))

    @test MD.is_valid_matrix_multiplication(A, x)
    @test MD.is_valid_matrix_multiplication(y, A)
    @test MD.is_valid_matrix_multiplication(A, z')
end

@testset "create BinaryOperation with matching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z")
    A = Tensor("A", Upper(1), Lower(2))

    @test typeof(A * x) == MD.BinaryOperation{*}
    @test typeof(y * A) == MD.BinaryOperation{*}
    @test typeof(z * A) == MD.BinaryOperation{*}
    @test typeof(A * z) == MD.BinaryOperation{*}
end

@testset "update_index column vector" begin
    x = Tensor("x", Upper(3))

    @test MD.update_index(x, Upper(3), Upper(3)) == x

    expected_shift = KrD(Lower(3), Upper(1))
    @test MD.update_index(x, Upper(3), Upper(1)) == MD.BinaryOperation{*}(x, expected_shift)

    expected_shift = KrD(Lower(3), Upper(2))
    @test MD.update_index(x, Upper(3), Upper(2)) == MD.BinaryOperation{*}(x, expected_shift)

    # update_index shall not transpose
    @test_throws DomainError MD.update_index(x, Upper(3), Lower(1))
end

@testset "update_index row vector" begin
    x = Tensor("x", Lower(3))

    @test MD.update_index(x, Lower(3), Lower(3)) == x

    expected_shift = KrD(Upper(3), Lower(1))
    @test MD.update_index(x, Lower(3), Lower(1)) == MD.BinaryOperation{*}(x, expected_shift)

    expected_shift = KrD(Upper(3), Lower(2))
    @test MD.update_index(x, Lower(3), Lower(2)) == MD.BinaryOperation{*}(x, expected_shift)

    # update_index shall not transpose
    @test_throws DomainError MD.update_index(x, Lower(3), Upper(1))
end

@testset "update_index matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    @test MD.update_index(A, Lower(2), Lower(2)) == A

    expected_shift = KrD(Upper(2), Lower(3))
    @test MD.update_index(A, Lower(2), Lower(3)) == MD.BinaryOperation{*}(A, expected_shift)

    # update_index shall not transpose
    @test_throws DomainError MD.update_index(A, Lower(2), Upper(3))
end

@testset "transpose vector" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    @test evaluate(x') == Tensor("x", Lower(1))
    @test evaluate(y') == Tensor("y", Upper(1))
end

@testset "combined update_index and transpose vector" begin
    x = Tensor("x", Upper(2))

    updated_transpose = evaluate(MD.update_index(x', Lower(2), Lower(1)))

    @test updated_transpose == Tensor("x", Lower(1))
end

@testset "transpose matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    A_transpose = evaluate(A')
    @test A_transpose == Tensor("A", Lower(1), Upper(2))
end

@testset "can_contract" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    z = Tensor("z", Lower(1))
    d = KrD(Lower(1), Lower(1))

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
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = A * x

    @test typeof(op1) == MD.BinaryOperation{*}
    @test MD.can_contract(op1.arg1, op1.arg2)
    @test typeof(op1.arg1) == MD.BinaryOperation{*}
    @test op1.arg2 == x

    op2 = x' * A

    @test typeof(op2) == MD.BinaryOperation{*}
    @test MD.can_contract(op2.arg1, op2.arg2)
    @test typeof(op2.arg1) == MD.BinaryOperation{*}
    @test typeof(op2.arg1.arg1) == MD.BinaryOperation{*}
    @test op2.arg1.arg1.arg1 == x
    @test op2.arg1.arg1.arg2 == KrD(Lower(3), Lower(3))
    @test op2.arg1.arg2 == KrD(Upper(3), Lower(1))
    @test op2.arg2 == A
end

@testset "create BinaryOperation with non-compatible matrix-vector fails" begin
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test_throws DomainError x * A
end

@testset "create BinaryOperation with non-matching indices vector-vector" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    op1 = x' * y

    @test typeof(op1) == MD.BinaryOperation{*}
    @test MD.can_contract(op1.arg1, op1.arg2)
    @test typeof(op1.arg1) == MD.BinaryOperation{*}
    @test op1.arg2 == y

    op2 = y' * x

    @test typeof(op2) == MD.BinaryOperation{*}
    @test MD.can_contract(op2.arg1, op2.arg2)
    @test typeof(op2.arg1) == MD.BinaryOperation{*}
    @test op2.arg2 == x
end

# TODO: Scalars not implemented
# @testset "create BinaryOperation with non-matching indices scalar-matrix" begin
#     A = Tensor("A", Upper(1), Lower(2))
#     z = Tensor("z")

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
#     x = Tensor("x", Upper(3))
#     z = Tensor("z")

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

@testset "to_string output is correct for primitive types" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(1), Upper(2), Upper(3), Lower(4), Upper(5), Lower(6), Lower(7))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z")
    d1 = KrD(Upper(1), Upper(1))
    d2 = KrD(Upper(3), Lower(4))
    zero = Zero(Upper(1), Lower(3), Lower(4))

    @test to_string(A) == "A¹₂"
    @test to_string(B) == "B¹²³₄⁵₆₇"
    @test to_string(x) == "x²"
    @test to_string(y) == "y₁"
    @test to_string(z) == "z"
    @test to_string(d1) == "δ¹¹"
    @test to_string(d2) == "δ³₄"
    @test to_string(zero) == "0¹₃₄"
end

@testset "to_string output is correct for BinaryOperation" begin
    a = Tensor("a")
    b = Tensor("b")

    mul = MD.BinaryOperation{*}(a, b)
    add = MD.BinaryOperation{+}(a, b)

    @test to_string(mul) == "ab"
    @test to_string(add) == "a + b"
    @test to_string(MD.BinaryOperation{+}(mul, b)) == "ab + b"
    @test to_string(MD.BinaryOperation{+}(mul, mul)) == "ab + ab"
    @test to_string(MD.BinaryOperation{*}(mul, mul)) == "abab"
    @test to_string(MD.BinaryOperation{*}(add, add)) == "(a + b)(a + b)"
end

@testset "to_std_string output is correct with matrix-vector contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, y')) == "yᵀA"
    @test to_std_string(contract(y', A)) == "yᵀA"
    @test to_std_string(contract(A', x')) == "xᵀAᵀ"
    @test to_std_string(contract(x', A')) == "xᵀAᵀ"
    @test to_std_string(contract(A', y)) == "Aᵀy"
    @test to_std_string(contract(y, A')) == "Aᵀy"
end

@testset "to_std_string output is correct with matrix-matrix contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))
    C = Tensor("C", Lower(1), Upper(3))
    D = Tensor("D", Lower(3), Upper(2))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, B)) == "AB"
    @test to_std_string(contract(B, A)) == "AB"
    @test to_std_string(contract(A, C)) == "CᵀA"
    @test to_std_string(contract(C, A)) == "CᵀA"
    @test to_std_string(contract(A, D)) == "ADᵀ"
    @test to_std_string(contract(D, A)) == "ADᵀ"
    @test to_std_string(contract(C, D)) == "(DC)ᵀ"
    @test to_std_string(contract(D, C)) == "(DC)ᵀ"
end
