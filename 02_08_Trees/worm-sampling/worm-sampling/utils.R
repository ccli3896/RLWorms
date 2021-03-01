# Utility functions
library(purrr)

# collapse position into a single index
# Goes from 1 ... n^dim
grid_to_index = function(pos, n, dim=2) {
    # pos is a vector (position)
    as.integer(sum((pos-1) * n^(seq(dim-1, 0))) + 1)
}

index_to_grid = function(idx, n, dim=2) {
    remainder = idx - 1L
    pos = integer(dim)
    for (i in seq_len(dim)) {
        divider = n^(dim - i)
        pos[i] = floor(remainder / divider) + 1
        remainder = remainder %% divider
    }
    pos
}
