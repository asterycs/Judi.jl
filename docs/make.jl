using Documenter
using DiffMatic

DocMeta.setdocmeta!(DiffMatic, :DocTestSetup, :(using DiffMatic); recursive = true)

makedocs(sitename = "DiffMatic", format = Documenter.HTML(), modules = [DiffMatic])

deploydocs(repo = "https://github.com/asterycs/DiffMatic.jl")
