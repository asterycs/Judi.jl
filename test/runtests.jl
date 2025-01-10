# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using Test

@testset "Aqua" begin
    include("Aqua.jl")
end

@testset "RicciTest" begin
    include("RicciTest.jl")
end

@testset "ForwardTest" begin
    include("ForwardTest.jl")
end

@testset "StdTest" begin
    include("StdTest.jl")
end
