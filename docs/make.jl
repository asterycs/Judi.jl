using Documenter
using Judi

DocMeta.setdocmeta!(Judi, :DocTestSetup, :(using Judi); recursive=true)

makedocs(
    sitename = "Judi",
    format = Documenter.HTML(),
    modules = [Judi]
)

deploydocs(
    repo = "https://github.com/asterycs/Judi.jl"
)
