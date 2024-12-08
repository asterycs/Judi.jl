using Yodi
using Test

yd = Yodi

@testset "Tensor constructor throws on invalid input" begin
    @test_throws DomainError Tensor("A", Lower(2), Lower(2))
    @test_throws DomainError Tensor("A", Lower(2), Upper(2), Lower(1), Lower(2))
end

@testset "Tensor constructor succeeds on valid input" begin
    @test !isnothing(Tensor("A", Upper(1), Lower(2)))
    @test !isnothing(Tensor("B", Lower(1), Lower(2)))
    @test !isnothing(Tensor("x", Upper(1)))
    @test !isnothing(Tensor("y", Lower(1)))
    @test !isnothing(Tensor("z"))
end

@testset "index equality operator" begin
    left = Lower(3)
    right = Lower(3)
    @test left == right
    @test left == Lower(3)
    @test left != Upper(3)
    @test left != Lower(1)
end

@testset "KrD constructor throws on invalid input" begin
    @test_throws DomainError KrD(Upper(2), Upper(2))
    @test_throws DomainError KrD(Lower(2), Lower(2))
    @test_throws DomainError KrD(Lower(1))
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
end

@testset "Zero constructor throws on invalid input" begin
    @test_throws DomainError Zero(Upper(2), Upper(2))
    @test_throws DomainError Zero(Lower(2), Lower(2))
    @test_throws DomainError Zero(Lower(1), Upper(2), Lower(1))
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

@testset "Sin constructor" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    op = sin(a * b)

    @test typeof(op) == yd.Sin
    @test typeof(op.arg) == yd.BinaryOperation{*}
end

@testset "Cos constructor" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    op = cos(a * b)

    @test typeof(op) == yd.Cos
    @test typeof(op.arg) == yd.BinaryOperation{*}
end

@testset "UnaryOperation equality operator" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    left = sin(a * b)

    @test left == yd.Sin(a * b)
    @test left != yd.Cos(a * b)
    @test left != -yd.Sin(a * b)
end

@testset "is_permutation true positive" begin
    l = [Lower(9); Upper(2); Lower(2); Lower(2)]
    r = collect(reverse(l))

    @test yd.is_permutation(l, l)
    @test yd.is_permutation(l, r)
    @test yd.is_permutation(r, l)
    @test yd.is_permutation(r, r)
end

@testset "is_permutation true negative" begin
    l = [Lower(9); Upper(2); Lower(2); Lower(2)]
    r1 = [Lower(9); Upper(2); Lower(2)]
    r2 = [Lower(9); Upper(2); Lower(2); Lower(2); Lower(2)]
    r3 = []
    r4 = [Lower(9); Upper(2); Upper(2); Lower(2)]

    @test !yd.is_permutation(l, r1)
    @test !yd.is_permutation(l, r2)
    @test !yd.is_permutation(l, r3)
    @test !yd.is_permutation(l, r4)
end

@testset "BinaryOperation equality operator" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = yd.BinaryOperation{*}(a, b)

    @test yd.BinaryOperation{*}(a, b) == yd.BinaryOperation{*}(a, b)
    @test left == yd.BinaryOperation{*}(a, b)
    @test left == yd.BinaryOperation{*}(b, a)
    @test left != yd.BinaryOperation{+}(a, b)
end

@testset "BinaryOperation equivalent" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = yd.BinaryOperation{*}(a, b)

    @test equivalent(yd.BinaryOperation{*}(a, b), yd.BinaryOperation{*}(a, b))
    @test equivalent(left, yd.BinaryOperation{*}(a, b))
    @test equivalent(left, yd.BinaryOperation{*}(b, a))
    @test !equivalent(left, yd.BinaryOperation{+}(a, b))
    @test !equivalent(left, yd.BinaryOperation{*}(a, Tensor("x", Upper(1))))
end

@testset "index hash function" begin
    @test hash(Lower(3)) == hash(Lower(3))
    @test hash(Lower(3)) != hash(Lower(1))
    @test hash(Lower(1)) != hash(Lower(3))
    @test hash(Upper(3)) != hash(Lower(3))
end

@testset "flip" begin
    @test yd.flip(Lower(3)) == Upper(3)
    @test yd.flip(Upper(3)) == Lower(3)
end

@testset "eliminate_indices removes correct indices" begin
    IdxUnion = yd.LowerOrUpperIndex

    indicesl = IdxUnion[
        Lower(9)
        Upper(9)
        Upper(3)
        Lower(2)
        Lower(1)
    ]

    indicesr = IdxUnion[
        Lower(3)
        Lower(2)
        Upper(3)
        Upper(9)
        Lower(9)
    ]

    l, r = yd.eliminate_indices(indicesl, indicesr)

    @test [l; r] == [Lower(2); Lower(1); Lower(2); Upper(3)]
    @test yd.eliminate_indices(IdxUnion[], IdxUnion[]) == (IdxUnion[], IdxUnion[])
end

@testset "eliminated_indices retains correct indices" begin
    IdxUnion = yd.LowerOrUpperIndex

    indicesl = IdxUnion[
        Lower(9)
        Upper(9)
        Upper(3)
        Lower(2)
        Lower(1)
    ]

    indicesr = IdxUnion[
        Lower(3)
        Lower(2)
        Upper(3)
        Upper(9)
        Lower(9)
    ]

    eliminated = yd.eliminated_indices(indicesl, indicesr)

    @test eliminated == IdxUnion[Lower(9); Upper(9); Upper(9); Lower(9); Upper(3); Lower(3)]
    @test yd.eliminated_indices(IdxUnion[], IdxUnion[]) == IdxUnion[]
end

@testset "get_free_indices with Tensor * Tensor and one matching pair" begin
    xt = Tensor("x", Lower(1)) # row vector
    A = Tensor("A", Upper(1), Lower(2))

    op1 = yd.BinaryOperation{*}(xt, A)
    op2 = yd.BinaryOperation{*}(A, xt)

    @test yd.get_free_indices(op1) == [Lower(2)]
    @test yd.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Tensor {+-} Tensor" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Lower(2), Upper(1))

    ops = (yd.BinaryOperation{+}, yd.BinaryOperation{-})

    for op ∈ ops
        op1 = op(A, A)
        op2 = op(A, B)
        op3 = op(B, A)

        @test yd.get_free_indices(op1) == [Upper(1); Lower(2)]
        @test yd.get_free_indices(op2) == [Upper(1); Lower(2)]
        @test yd.get_free_indices(op3) == [Lower(2); Upper(1)]
    end
end

@testset "get_free_indices with Tensor * KrD and one matching pair" begin
    x = Tensor("x", Upper(1))
    δ = KrD(Lower(1), Lower(2))

    op1 = yd.BinaryOperation{*}(x, δ)
    op2 = yd.BinaryOperation{*}(δ, x)

    @test yd.get_free_indices(op1) == [Lower(2)]
    @test yd.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with scalar Tensor * KrD" begin
    x = Tensor("x")
    δ = KrD(Lower(1), Lower(2))

    op1 = yd.BinaryOperation{*}(x, δ)
    op2 = yd.BinaryOperation{*}(δ, x)

    @test yd.get_free_indices(op1) == [Lower(1); Lower(2)]
    @test yd.get_free_indices(op2) == [Lower(1); Lower(2)]
end

@testset "get_free_indices with Tensor * Tensor and no matching pairs" begin
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = yd.BinaryOperation{*}(x, A)
    op2 = yd.BinaryOperation{*}(A, x)

    @test yd.get_free_indices(op1) == [Upper(1); Upper(1); Lower(2)]
    @test yd.get_free_indices(op2) == [Upper(1); Lower(2); Upper(1)]
end

@testset "is_contraction_unambigous vector * vector with matching pair" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    @test yd.is_contraction_unambigous(x, y)
    @test yd.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous vector * vector with non-matching pair" begin
    x = Tensor("x", Lower(1))
    y = Tensor("y", Lower(1))

    @test !yd.is_contraction_unambigous(x, y)
    @test !yd.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous matrix * vector with matching pair" begin
    x = Tensor("x", Upper(2))
    A = Tensor("A", Upper(1), Lower(2))

    @test yd.is_contraction_unambigous(A, x)
    @test yd.is_contraction_unambigous(x, A)
end

@testset "is_contraction_unambigous matrix * vector with non-matching pair" begin
    x = Tensor("x", Lower(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test yd.is_contraction_unambigous(A, x)
    @test yd.is_contraction_unambigous(x, A)
end

@testset "is_valid_matrix_multiplication fails with invalid input" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(2))

    @test !yd.is_valid_matrix_multiplication(A, x)
    @test !yd.is_valid_matrix_multiplication(x, A)
    @test !yd.is_valid_matrix_multiplication(A, y)
    @test !yd.is_valid_matrix_multiplication(y, A)
end

@testset "is_valid_matrix_multiplication succeeds with valid input" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z", Lower(2))

    @test yd.is_valid_matrix_multiplication(A, x)
    @test yd.is_valid_matrix_multiplication(y, A)
    @test yd.is_valid_matrix_multiplication(x, z)
end

@testset "multiplication with matching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test A * x == yd.BinaryOperation{*}(A, x)
    @test y * A == yd.BinaryOperation{*}(y, A)
    @test A * B == yd.BinaryOperation{*}(A, B)
end

@testset "multiplication with ambigous input fails" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z", Upper(1))
    A = Tensor("A", Upper(1), Lower(2), Lower(3))
    B = Tensor("B", Upper(1), Lower(2))

    @test_throws DomainError A * x
    @test_throws DomainError y * A
    @test_throws DomainError A * A
    @test_throws DomainError B * z
end

@testset "multiplication with scalars" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    A = Tensor("A", Upper(1), Lower(2), Lower(3))
    z = Tensor("z")
    r = 42

    for n ∈ (z, r)
        for t ∈ (x, y, A, z)
            @test n * t == yd.BinaryOperation{*}(n, t)
            @test t * n == yd.BinaryOperation{*}(t, n)
        end
    end
end

@testset "elementwise multiplication matrix-matrix" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))

    op1 = A .* A

    @test typeof(op1) == yd.BinaryOperation{*}
    @test equivalent(evaluate(op1.arg1), Tensor("A", Upper(1), Lower(2)))
    @test equivalent(evaluate(op1.arg2), Tensor("A", Upper(1), Lower(2)))

    op2 = A .* B

    @test typeof(op2) == yd.BinaryOperation{*}
    @test equivalent(evaluate(op2.arg1), Tensor("A", Upper(3), Lower(4)))
    @test equivalent(evaluate(op2.arg2), Tensor("B", Upper(3), Lower(4)))

    op3 = A' .* B'

    @test typeof(op3) == yd.BinaryOperation{*}
    @test equivalent(evaluate(op3.arg1), Tensor("A", Lower(1), Upper(2)))
    @test equivalent(evaluate(op3.arg2), Tensor("B", Lower(3), Upper(4)))
end

@testset "elementwise multiplication vector-vector" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))

    op1 = x .* x

    @test typeof(op1) == yd.BinaryOperation{*}
    @test equivalent(evaluate(op1.arg1), Tensor("x", Upper(1)))
    @test equivalent(evaluate(op1.arg2), Tensor("x", Upper(1)))

    op2 = x .* y

    @test typeof(op2) == yd.BinaryOperation{*}
    @test equivalent(evaluate(op2.arg1), Tensor("x", Upper(2)))
    @test equivalent(evaluate(op2.arg2), Tensor("y", Upper(2)))

    op3 = x' .* y'

    @test typeof(op3) == yd.BinaryOperation{*}
    @test equivalent(evaluate(op3.arg1), Tensor("x", Lower(2)))
    @test equivalent(evaluate(op3.arg2), Tensor("y", Lower(2)))
end

@testset "elementwise multiplication with ambiguous input fails" begin
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(3), Lower(4))
    B = Tensor("B", Upper(5), Upper(6))
    T = Tensor("T", Upper(7), Lower(8), Lower(9))

    @test_throws DomainError x .* x'
    @test_throws DomainError x' .* x
    @test_throws DomainError x .* A
    @test_throws DomainError A .* x
    @test_throws DomainError A .* x'
    @test_throws DomainError x' .* A
    @test_throws DomainError A .* B
    @test_throws DomainError B .* A
    @test_throws DomainError A .* T
    @test_throws DomainError T .* A
end

@testset "update_index column vector" begin
    x = Tensor("x", Upper(3))

    @test yd.update_index(x, Upper(3), Upper(3)) == x

    expected_shift = KrD(Lower(3), Upper(1))
    @test yd.update_index(x, Upper(3), Upper(1)) == yd.BinaryOperation{*}(x, expected_shift)

    expected_shift = KrD(Lower(3), Upper(2))
    @test yd.update_index(x, Upper(3), Upper(2)) == yd.BinaryOperation{*}(x, expected_shift)
end

@testset "update_index row vector" begin
    x = Tensor("x", Lower(3))

    @test yd.update_index(x, Lower(3), Lower(3)) == x

    expected_shift = KrD(Upper(3), Lower(1))
    @test yd.update_index(x, Lower(3), Lower(1)) == yd.BinaryOperation{*}(x, expected_shift)

    expected_shift = KrD(Upper(3), Lower(2))
    @test yd.update_index(x, Lower(3), Lower(2)) == yd.BinaryOperation{*}(x, expected_shift)
end

@testset "update_index matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    @test yd.update_index(A, Lower(2), Lower(2)) == A

    expected_shift = KrD(Upper(2), Lower(3))
    @test yd.update_index(A, Lower(2), Lower(3)) == yd.BinaryOperation{*}(A, expected_shift)
end

@testset "transpose vector" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    @test equivalent(evaluate(x'), Tensor("x", Lower(1)))
    @test equivalent(evaluate(y'), Tensor("y", Upper(1)))
end

@testset "transpose KrD" begin
    d = KrD(Upper(1), Lower(2))

    @test equivalent(evaluate(d'), KrD(Lower(1), Upper(2)))
end

@testset "combined update_index and transpose vector" begin
    x = Tensor("x", Upper(2))

    xt = x'
    x_indices = yd.get_free_indices(xt)
    updated_transpose = evaluate(yd.update_index(xt, x_indices[1], Lower(1)))

    @test equivalent(updated_transpose, Tensor("x", Lower(1)))
end

@testset "transpose unary operations" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    ops = (sin, cos)
    types = (yd.Sin, yd.Cos)

    for (op, type) ∈ zip(ops, types)
        op1 = evaluate(op(y * x)')
        op2 = evaluate(op(x)')

        @test typeof(op1) == type
        @test evaluate(op1.arg) == y * x
        @test equivalent(evaluate(op2).arg, Tensor("x", Lower(1)))
    end
end

@testset "negate any operation" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    ops = (A, x, A * x, A + A, sin(x), cos(x), tr(A))

    for op ∈ ops
        @test typeof(-op) == yd.Negate
        @test (-op).arg == op
    end
end

@testset "transpose matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    At = evaluate(A')
    @test equivalent(At, Tensor("A", Lower(1), Upper(2)))
end

@testset "transpose BinaryOperation{*}" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))

    op_t = evaluate((A * x)')
    @test equivalent(
        evaluate(op_t),
        yd.BinaryOperation{*}(Tensor("A", Lower(1), Lower(2)), Tensor("x", Upper(2))),
    )
end

@testset "add/subtract tensors with different order fails" begin
    a = Tensor("a")
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(1), Lower(2))
    T = Tensor("T", Upper(1), Lower(2), Lower(3))

    tensors = (a, x, A, T)

    for l ∈ tensors
        for r ∈ tensors
            if l == r
                continue
            end

            @test_throws DomainError l + r
            @test_throws DomainError l - r
        end
    end
end

# TODO: Triggers assertion for now, need to think how to update
# @testset "add/subtract tensors with ambiguous indices fails" begin
#     A = Tensor("A", Upper(1), Lower(2))
#     B = Tensor("B", Upper(2), Lower(3))

#     @test_throws DomainError A + B
# end

@testset "add/subtract tensors with different indices" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))

    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))


    for op ∈ (+, -)
        for args ∈ ((x, y), (A, B))
            e = evaluate((op(args[1], args[2])))
            @test e.arg1.indices == e.arg2.indices
        end
    end
end

@testset "transpose BinaryOperation{+-}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))
    z = Tensor("z", Upper(3))

    for op ∈ (+, -)
        for ags ∈ ((x, y), (x, z))
            op_t = evaluate((op(x, y))')
            @test op_t.arg1.indices == op_t.arg2.indices
        end
    end
end

@testset "trace with matrix input works" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test isempty(yd.get_free_indices(tr(A)))
    @test isempty(yd.get_free_indices(tr(A * B)))
end

@testset "trace with non-matrix input fails" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))
    x = Tensor("x", Upper(3))

    @test_throws DomainError tr(x)
    @test_throws DomainError tr(A * x)
    @test_throws DomainError tr(x + x)
    @test_throws DomainError tr(A * B * x)
end

@testset "can_contract" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    z = Tensor("z", Lower(1))
    d = KrD(Lower(1), Upper(3))

    @test yd.can_contract(A, x)
    @test yd.can_contract(x, A)
    @test !yd.can_contract(A, y)
    @test !yd.can_contract(y, A)
    @test yd.can_contract(A, d)
    @test yd.can_contract(d, A)
    @test yd.can_contract(A, z)
    @test yd.can_contract(z, A)
end

@testset "multiplication with non-matching indices matrix-vector" begin
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = A * x

    @test typeof(op1) == yd.BinaryOperation{*}
    @test yd.can_contract(op1.arg1, op1.arg2)
    @test typeof(op1.arg1) == yd.BinaryOperation{*}
    @test op1.arg2 == x

    op2 = x' * A

    @test typeof(op2) == yd.BinaryOperation{*}
    @test yd.can_contract(op2.arg1, op2.arg2)
    @test typeof(op2.arg1) == yd.BinaryOperation{*}
    @test typeof(op2.arg1.arg1) == yd.Adjoint
    @test typeof(op2.arg1.arg1.expr) == yd.BinaryOperation{*}
    @test op2.arg1.arg1.expr.arg1 == x
    @test typeof(op2.arg1.arg1.expr.arg2) == KrD
    @test op2.arg1.arg1.expr.arg2.indices[1] == yd.flip(x.indices[1])
    @test typeof(op2.arg1.arg2) == KrD
    @test yd.flip(op2.arg1.arg2.indices[1]) == op2.arg1.arg1.expr.arg2.indices[2]
    @test yd.flip(op2.arg1.arg2.indices[2]) == A.indices[1]
    @test op2.arg2 == A
end

@testset "multiplication with non-compatible matrix-vector fails" begin
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test_throws DomainError x * A
end

@testset "multiplication with adjoint matrix-matrix has correct indices" begin
    A = Tensor("A", Upper(1), Lower(2))
    C = Tensor("C", Upper(3), Lower(4))

    ops = (A' * C, A' * C')

    # TODO: Check also efter evaluate
    for op ∈ ops
        @assert typeof(op) == yd.BinaryOperation{*}
        op_indices = yd.get_free_indices(op)
        @test length(op_indices) == 2

        @test typeof(yd.get_free_indices(op.arg1)[1]) == Lower
        @test yd.flip(yd.get_free_indices(op.arg1)[1]) == yd.get_free_indices(op.arg2)[1]
        @test typeof(yd.get_free_indices(op.arg2)[end]) == Lower
    end
end

@testset "multiplication with ambiguous indices throws" begin
    A = Tensor("A", Upper(2), Lower(1))
    At = Tensor("A", Upper(1), Lower(2)) # manual transpose since adjoint updates the indices
    B = Tensor("B", Upper(2), Lower(3))
    Bt = Tensor("B", Upper(3), Lower(2))
    d = KrD(Upper(2), Lower(2))
    x = Tensor("x", Upper(2))

    @test_throws DomainError A * B
    @test_throws DomainError At * Bt
    @test_throws DomainError A * Bt
    @test_throws DomainError x * d
    @test_throws DomainError d * x
end

@testset "vector inner product with mismatching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    op1 = x' * y

    @test typeof(op1) == yd.BinaryOperation{*}
    @test yd.can_contract(op1.arg1, op1.arg2)
    @test typeof(op1.arg1) == yd.BinaryOperation{*}
    @test op1.arg2 == y

    op2 = y' * x

    @test typeof(op2) == yd.BinaryOperation{*}
    @test yd.can_contract(op2.arg1, op2.arg2)
    @test typeof(op2.arg1) == yd.BinaryOperation{*}
    @test op2.arg2 == x
end

@testset "multiplication with non-matching indices scalar-matrix" begin
    A = Tensor("A", Upper(1), Lower(2))
    z = Tensor("z")

    op1 = A * z
    op2 = z * A

    @test typeof(op1) == yd.BinaryOperation{*}
    @test !yd.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == A
    @test op1.arg2 == z

    @test typeof(op2) == yd.BinaryOperation{*}
    @test !yd.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == z
    @test op2.arg2 == A
end

@testset "multiplication with non-matching indices scalar-vector" begin
    x = Tensor("x", Upper(3))
    z = Tensor("z")

    op1 = z * x
    op2 = x * z

    @test typeof(op1) == yd.BinaryOperation{*}
    @test !yd.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == z
    @test op1.arg2 == x

    @test typeof(op2) == yd.BinaryOperation{*}
    @test !yd.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == x
    @test op2.arg2 == z
end

@testset "to_string output is correct for primitive types" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(1), Upper(2), Upper(3), Lower(4), Upper(5), Lower(6), Lower(7))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z")
    d1 = KrD(Upper(1), Upper(2))
    d2 = KrD(Upper(3), Lower(4))
    zero = Zero(Upper(1), Lower(3), Lower(4))

    @test to_string(A) == "A¹₂"
    @test to_string(B) == "B¹²³₄⁵₆₇"
    @test to_string(x) == "x²"
    @test to_string(y) == "y₁"
    @test to_string(z) == "z"
    @test to_string(d1) == "δ¹²"
    @test to_string(d2) == "δ³₄"
    @test to_string(zero) == "0¹₃₄"
end

@testset "to_string output is correct for BinaryOperation" begin
    a = Tensor("a")
    b = Tensor("b")

    mul = yd.BinaryOperation{*}(a, b)
    add = yd.BinaryOperation{+}(a, b)
    sub = yd.BinaryOperation{-}(a, b)

    @test to_string(mul) == "ab"
    @test to_string(add) == "a + b"
    @test to_string(sub) == "a - b"
    @test to_string(yd.BinaryOperation{+}(mul, b)) == "ab + b"
    @test to_string(yd.BinaryOperation{+}(mul, mul)) == "ab + ab"
    @test to_string(yd.BinaryOperation{-}(mul, mul)) == "ab - ab"
    @test to_string(yd.BinaryOperation{*}(mul, mul)) == "abab"
    @test to_string(yd.BinaryOperation{*}(add, add)) == "(a + b)(a + b)"
    @test to_string(yd.BinaryOperation{*}(sub, add)) == "(a - b)(a + b)"
end
