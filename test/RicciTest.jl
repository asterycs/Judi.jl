# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using Judi
using Test

jd = Judi

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

    @test typeof(op) == jd.Sin
    @test typeof(op.arg) == jd.BinaryOperation{jd.Mult}
end

@testset "Cos constructor" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    op = cos(a * b)

    @test typeof(op) == jd.Cos
    @test typeof(op.arg) == jd.BinaryOperation{jd.Mult}
end

@testset "UnaryOperation equality operator" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    inner = a * b

    left = sin(inner)

    @test left == jd.Sin(inner)
    @test left != jd.Cos(inner)
    @test left != -jd.Sin(inner)
end

@testset "is_permutation true positive" begin
    l = [Lower(9); Upper(2); Lower(2); Lower(2)]
    r = collect(reverse(l))

    @test jd.is_permutation(l, l)
    @test jd.is_permutation(l, r)
    @test jd.is_permutation(r, l)
    @test jd.is_permutation(r, r)
end

@testset "is_permutation true negative" begin
    l = [Lower(9); Upper(2); Lower(2); Lower(2)]
    r1 = [Lower(9); Upper(2); Lower(2)]
    r2 = [Lower(9); Upper(2); Lower(2); Lower(2); Lower(2)]
    r3 = []
    r4 = [Lower(9); Upper(2); Upper(2); Lower(2)]

    @test !jd.is_permutation(l, r1)
    @test !jd.is_permutation(l, r2)
    @test !jd.is_permutation(l, r3)
    @test !jd.is_permutation(l, r4)
end

@testset "BinaryOperation equality operator" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = jd.BinaryOperation{jd.Mult}(a, b)

    @test jd.BinaryOperation{jd.Mult}(a, b) == jd.BinaryOperation{jd.Mult}(a, b)
    @test left == jd.BinaryOperation{jd.Mult}(a, b)
    @test left == jd.BinaryOperation{jd.Mult}(b, a)
    @test left != jd.BinaryOperation{jd.Add}(a, b)
end

@testset "BinaryOperation equivalent" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = jd.BinaryOperation{jd.Mult}(a, b)

    @test equivalent(jd.BinaryOperation{jd.Mult}(a, b), jd.BinaryOperation{jd.Mult}(a, b))
    @test equivalent(left, jd.BinaryOperation{jd.Mult}(a, b))
    @test equivalent(left, jd.BinaryOperation{jd.Mult}(b, a))
    @test !equivalent(left, jd.BinaryOperation{jd.Add}(a, b))
    @test !equivalent(left, jd.BinaryOperation{jd.Mult}(a, Tensor("x", Upper(1))))
end

@testset "index hash function" begin
    @test hash(Lower(3)) == hash(Lower(3))
    @test hash(Lower(3)) != hash(Lower(1))
    @test hash(Lower(1)) != hash(Lower(3))
    @test hash(Upper(3)) != hash(Lower(3))
end

@testset "flip" begin
    @test jd.flip(Lower(3)) == Upper(3)
    @test jd.flip(Upper(3)) == Lower(3)
end

@testset "eliminate_indices removes correct indices" begin
    IdxUnion = jd.LowerOrUpperIndex

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

    l, r = jd.eliminate_indices(indicesl, indicesr)

    @test [l; r] == [Lower(2); Lower(1); Lower(2); Upper(3)]
    @test jd.eliminate_indices(IdxUnion[], IdxUnion[]) == (IdxUnion[], IdxUnion[])
end

@testset "eliminated_indices retains correct indices" begin
    IdxUnion = jd.LowerOrUpperIndex

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

    eliminated = jd.eliminated_indices(indicesl, indicesr)

    @test eliminated == IdxUnion[Lower(9); Upper(9); Upper(9); Lower(9); Upper(3); Lower(3)]
    @test jd.eliminated_indices(IdxUnion[], IdxUnion[]) == IdxUnion[]
end

@testset "get_free_indices with Tensor * Tensor and one matching pair" begin
    xt = Tensor("x", Lower(1)) # row vector
    A = Tensor("A", Upper(1), Lower(2))

    op1 = jd.BinaryOperation{jd.Mult}(xt, A)
    op2 = jd.BinaryOperation{jd.Mult}(A, xt)

    @test jd.get_free_indices(op1) == [Lower(2)]
    @test jd.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Tensor {+-} Tensor" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Lower(2), Upper(1))

    ops = (jd.BinaryOperation{jd.Add}, jd.BinaryOperation{jd.Sub})

    for op ∈ ops
        op1 = op(A, A)
        op2 = op(A, B)
        op3 = op(B, A)

        @test jd.get_free_indices(op1) == [Upper(1); Lower(2)]
        @test jd.get_free_indices(op2) == [Upper(1); Lower(2)]
        @test jd.get_free_indices(op3) == [Lower(2); Upper(1)]
    end
end

@testset "get_free_indices with Tensor * KrD and one matching pair" begin
    x = Tensor("x", Upper(1))
    δ = KrD(Lower(1), Lower(2))

    op1 = jd.BinaryOperation{jd.Mult}(x, δ)
    op2 = jd.BinaryOperation{jd.Mult}(δ, x)

    @test jd.get_free_indices(op1) == [Lower(2)]
    @test jd.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with scalar Tensor * KrD" begin
    x = Tensor("x")
    δ = KrD(Lower(1), Lower(2))

    op1 = jd.BinaryOperation{jd.Mult}(x, δ)
    op2 = jd.BinaryOperation{jd.Mult}(δ, x)

    @test jd.get_free_indices(op1) == [Lower(1); Lower(2)]
    @test jd.get_free_indices(op2) == [Lower(1); Lower(2)]
end

@testset "get_free_indices with Tensor * Tensor and no matching pairs" begin
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = jd.BinaryOperation{jd.Mult}(x, A)
    op2 = jd.BinaryOperation{jd.Mult}(A, x)

    @test jd.get_free_indices(op1) == [Upper(1); Upper(1); Lower(2)]
    @test jd.get_free_indices(op2) == [Upper(1); Lower(2); Upper(1)]
end

@testset "is_contraction_unambigous vector * vector with matching pair" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    @test jd.is_contraction_unambigous(x, y)
    @test jd.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous vector * vector with non-matching pair" begin
    x = Tensor("x", Lower(1))
    y = Tensor("y", Lower(1))

    @test !jd.is_contraction_unambigous(x, y)
    @test !jd.is_contraction_unambigous(y, x)
end

@testset "is_contraction_unambigous matrix * vector with matching pair" begin
    x = Tensor("x", Upper(2))
    A = Tensor("A", Upper(1), Lower(2))

    @test jd.is_contraction_unambigous(A, x)
    @test jd.is_contraction_unambigous(x, A)
end

@testset "is_contraction_unambigous matrix * vector with non-matching pair" begin
    x = Tensor("x", Lower(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test jd.is_contraction_unambigous(A, x)
    @test jd.is_contraction_unambigous(x, A)
end

@testset "multiplication with matching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test jd.get_free_indices(A * x) == jd.get_free_indices(jd.BinaryOperation{jd.Mult}(A, x))
    @test jd.get_free_indices(y * A) == jd.get_free_indices(jd.BinaryOperation{jd.Mult}(y, A))
    @test jd.get_free_indices(A * B) == jd.get_free_indices(jd.BinaryOperation{jd.Mult}(A, B))
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
end

@testset "multiplication with scalars" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    A = Tensor("A", Upper(1), Lower(2), Lower(3))
    z = Tensor("z")
    r = 42

    for n ∈ (z, r)
        for t ∈ (x, y, A, z)
            @test n * t == jd.BinaryOperation{jd.Mult}(n, t)
            @test t * n == jd.BinaryOperation{jd.Mult}(t, n)
        end
    end
end

@testset "elementwise multiplication matrix-matrix" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))

    op1 = A .* A

    @test typeof(op1) == jd.BinaryOperation{jd.Mult}
    @test equivalent(evaluate(op1.arg1), Tensor("A", Upper(1), Lower(2)))
    @test equivalent(evaluate(op1.arg2), Tensor("A", Upper(1), Lower(2)))

    op2 = A .* B

    @test typeof(op2) == jd.BinaryOperation{jd.Mult}
    @test equivalent(evaluate(op2.arg1), Tensor("A", Upper(3), Lower(4)))
    @test equivalent(evaluate(op2.arg2), Tensor("B", Upper(3), Lower(4)))

    op3 = A' .* B'

    @test typeof(op3) == jd.BinaryOperation{jd.Mult}
    @test equivalent(evaluate(op3.arg1), Tensor("A", Lower(1), Upper(2)))
    @test equivalent(evaluate(op3.arg2), Tensor("B", Lower(3), Upper(4)))
end

@testset "elementwise multiplication vector-vector" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))

    op1 = x .* x

    @test typeof(op1) == jd.BinaryOperation{jd.Mult}
    @test equivalent(evaluate(op1.arg1), Tensor("x", Upper(1)))
    @test equivalent(evaluate(op1.arg2), Tensor("x", Upper(1)))

    op2 = x .* y

    @test typeof(op2) == jd.BinaryOperation{jd.Mult}
    @test equivalent(evaluate(op2.arg1), Tensor("x", Upper(2)))
    @test equivalent(evaluate(op2.arg2), Tensor("y", Upper(2)))

    op3 = x' .* y'

    @test typeof(op3) == jd.BinaryOperation{jd.Mult}
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

    @test jd.update_index(x, Upper(3), Upper(3)) == x

    expected_shift = KrD(Lower(3), Upper(1))
    @test jd.update_index(x, Upper(3), Upper(1)) == jd.BinaryOperation{jd.Mult}(x, expected_shift)

    expected_shift = KrD(Lower(3), Upper(2))
    @test jd.update_index(x, Upper(3), Upper(2)) == jd.BinaryOperation{jd.Mult}(x, expected_shift)
end

@testset "update_index row vector" begin
    x = Tensor("x", Lower(3))

    @test jd.update_index(x, Lower(3), Lower(3)) == x

    expected_shift = KrD(Upper(3), Lower(1))
    @test jd.update_index(x, Lower(3), Lower(1)) == jd.BinaryOperation{jd.Mult}(x, expected_shift)

    expected_shift = KrD(Upper(3), Lower(2))
    @test jd.update_index(x, Lower(3), Lower(2)) == jd.BinaryOperation{jd.Mult}(x, expected_shift)
end

@testset "update_index matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    @test jd.update_index(A, Lower(2), Lower(2)) == A

    expected_shift = KrD(Upper(2), Lower(3))
    @test jd.update_index(A, Lower(2), Lower(3)) == jd.BinaryOperation{jd.Mult}(A, expected_shift)
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
    x_indices = jd.get_free_indices(xt)
    updated_transpose = evaluate(jd.update_index(xt, x_indices[1], Lower(1)))

    @test equivalent(updated_transpose, Tensor("x", Lower(1)))
end

@testset "transpose unary operations" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    ops = (sin, cos)
    types = (jd.Sin, jd.Cos)

    for (op, type) ∈ zip(ops, types)
        op1 = evaluate(op(y * x)')
        op2 = evaluate(op(x)')

        @test typeof(op1) == type
        @test jd.get_free_indices(op1.arg) == jd.get_free_indices(y * x)
        @test equivalent(evaluate(op2).arg, Tensor("x", Lower(1)))
    end
end

@testset "negate any operation" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    ops = (A, x, A * x, A + A, sin(x), cos(x), tr(A))

    for op ∈ ops
        @test typeof(-op) == jd.Negate
        @test (-op).arg == op
    end
end

@testset "transpose matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    At = evaluate(A')
    @test equivalent(At, Tensor("A", Lower(1), Upper(2)))
end

@testset "transpose BinaryOperation{jd.Mult}" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))

    op_t = evaluate((A * x)')
    @test equivalent(
        evaluate(op_t),
        jd.BinaryOperation{jd.Mult}(Tensor("A", Lower(1), Lower(2)), Tensor("x", Upper(2))),
    )
end

@testset "jd.Add/jd.Subtract tensors with different order fails" begin
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

@testset "jd.Add/jd.Subtract tensors with ambiguous indices succeeds" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test equivalent(evaluate(A + B), jd.BinaryOperation{jd.Add}(Tensor("A", Upper(1), Lower(2)), Tensor("B", Upper(1), Lower(2))))
end

@testset "jd.Add/jd.Subtract tensors with different indices" begin
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

    @test isempty(jd.get_free_indices(tr(A)))
    @test isempty(jd.get_free_indices(tr(A * B)))
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

    @test jd.can_contract(A, x)
    @test jd.can_contract(x, A)
    @test !jd.can_contract(A, y)
    @test !jd.can_contract(y, A)
    @test jd.can_contract(A, d)
    @test jd.can_contract(d, A)
    @test jd.can_contract(A, z)
    @test jd.can_contract(z, A)
end

@testset "multiplication with non-matching indices matrix-vector" begin
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = A * x

    @test typeof(op1) == jd.BinaryOperation{jd.Mult}
    @test jd.can_contract(op1.arg1, op1.arg2)
    @test jd.get_free_indices(op1) == jd.LowerOrUpperIndex[Upper(1)]

    op2 = x' * A

    @test typeof(op2) == jd.BinaryOperation{jd.Mult}
    @test jd.can_contract(op2.arg1, op2.arg2)
    @test jd.get_free_indices(op2) == jd.LowerOrUpperIndex[Lower(2)]
end

@testset "multiplication with non-compatible matrix-vector fails" begin
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test_throws DomainError x * A
end

@testset "multiplication with matrix'-matrix has correct indices" begin
    A = Tensor("A", Upper(1), Lower(2))
    C = Tensor("C", Upper(3), Lower(4))

    op = A' * C

    @assert typeof(op) == jd.BinaryOperation{jd.Mult}
    op_indices = jd.get_free_indices(op)
    @test length(op_indices) == 2

    @test typeof(jd.get_free_indices(op.arg1)[1]) == Lower
    @test jd.flip(jd.get_free_indices(op.arg1)[1]) == jd.get_free_indices(op.arg2)[1]
    @test typeof(jd.get_free_indices(op.arg2)[end]) == Lower
end

@testset "multiplication with matrix'-matrix' has correct indices" begin
    A = Tensor("A", Upper(1), Lower(2))
    C = Tensor("C", Upper(3), Lower(4))

    op = A' * C'

    @assert typeof(op) == jd.BinaryOperation{jd.Mult}
    op_indices = jd.get_free_indices(op)
    @test length(op_indices) == 2

    @test typeof(jd.get_free_indices(op.arg1)[1]) == Lower
    @test jd.flip(jd.get_free_indices(op.arg1)[1]) == jd.get_free_indices(op.arg2)[2]
    @test typeof(jd.get_free_indices(op.arg2)[end]) == Upper
end

@testset "vector inner product with mismatching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    op1 = x' * y

    @test typeof(op1) == jd.BinaryOperation{jd.Mult}
    @test jd.can_contract(op1.arg1, op1.arg2)
    @test isempty(jd.get_free_indices(op1))

    op2 = y' * x

    @test typeof(op2) == jd.BinaryOperation{jd.Mult}
    @test jd.can_contract(op2.arg1, op2.arg2)
    @test isempty(jd.get_free_indices(op2))
end

@testset "multiplication with non-matching indices scalar-matrix" begin
    A = Tensor("A", Upper(1), Lower(2))
    z = Tensor("z")

    op1 = A * z
    op2 = z * A

    @test typeof(op1) == jd.BinaryOperation{jd.Mult}
    @test !jd.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == A
    @test op1.arg2 == z

    @test typeof(op2) == jd.BinaryOperation{jd.Mult}
    @test !jd.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == z
    @test op2.arg2 == A
end

@testset "multiplication with non-matching indices scalar-vector" begin
    x = Tensor("x", Upper(3))
    z = Tensor("z")

    op1 = z * x
    op2 = x * z

    @test typeof(op1) == jd.BinaryOperation{jd.Mult}
    @test !jd.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == z
    @test op1.arg2 == x

    @test typeof(op2) == jd.BinaryOperation{jd.Mult}
    @test !jd.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == x
    @test op2.arg2 == z
end

@testset "multiplication with adjoint and adjoint of multiplication is equal" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test jd.get_free_indices(x' * A') == jd.get_free_indices((A * x)')
    @test jd.get_free_indices(x' * A) == jd.get_free_indices((A' * x)')
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

    mul = jd.BinaryOperation{jd.Mult}(a, b)
    add = jd.BinaryOperation{jd.Add}(a, b)
    sub = jd.BinaryOperation{jd.Sub}(a, b)

    @test to_string(mul) == "ab"
    @test to_string(add) == "a + b"
    @test to_string(sub) == "a - b"
    @test to_string(jd.BinaryOperation{jd.Add}(mul, b)) == "ab + b"
    @test to_string(jd.BinaryOperation{jd.Add}(mul, mul)) == "ab + ab"
    @test to_string(jd.BinaryOperation{jd.Sub}(mul, mul)) == "ab - ab"
    @test to_string(jd.BinaryOperation{jd.Mult}(mul, mul)) == "abab"
    @test to_string(jd.BinaryOperation{jd.Mult}(add, add)) == "(a + b)(a + b)"
    @test to_string(jd.BinaryOperation{jd.Mult}(sub, add)) == "(a - b)(a + b)"
end
