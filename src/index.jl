# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import Base.hash

export Upper, Lower

Letter = Int

struct Upper
    letter::Letter
end

struct Lower
    letter::Letter
end

LowerOrUpperIndex = Union{Lower,Upper}
IndexList = Vector{LowerOrUpperIndex}

function flip(index::Lower)
    return Upper(index.letter)
end

function flip(index::Upper)
    return Lower(index.letter)
end

function flip_to(index::Lower, letter::Letter)
    return Upper(letter)
end

function flip_to(index::Upper, letter::Letter)
    return Lower(letter)
end

function same_to(old::Lower, letter::Letter)
    return Lower(letter)
end

function same_to(old::Upper, letter::Letter)
    return Upper(letter)
end

function hash(arg::LowerOrUpperIndex)
    h::UInt = 0

    if typeof(arg) == Lower
        h |= 1
    end

    h |= arg.letter << 1

    h
end

function get_unique_indices(arg::IndexList)
    with_counts = [(i => count(==(i), arg)) for i ∈ unique(arg)]

    return [i for (i,c) ∈ with_counts if c == 1]
end

function get_repeated_indices(arg::IndexList)
    with_counts = [(i => count(==(i), arg)) for i ∈ unique(arg)]

    return [i for (i,c) ∈ with_counts if c != 1]
end
