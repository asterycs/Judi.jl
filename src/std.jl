function create_matrix(name::String)
    return Sym(name, Upper(get_next_letter()), Lower(get_next_letter()))
end

function create_vector(name::String)
    return Sym(name, Upper(get_next_letter()))
end