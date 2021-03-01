# Data-generating model for simulations

library(tidyverse)
source("utils.R")

# response surface for 'light off' and 'light on'
resp_off = function(x, n) 2*x[1]/n + 2*(x[2]/n)^2
resp_on = function(x, n) 2*(x[1]/n)^3 + 5*(x[2]/n) - 1.5

# worm spends its time mostly in a vertical strip
# marginal worm preference
move_pref = function(x, n)  dbeta(x / (n + 1), 3, 3)

# simulate worm motion in `n`^`dim` grid, according to worm preferences,
# and sample complete data + noise
# worm starts in center of grid by default (`init`)
sample_worm = function(n=12, dim=2, steps=100, error=1,
                       init=rep(as.integer(n/2), dim)) {
    on_samp = numeric(steps)
    off_samp = numeric(steps)
    pos = list(init)

    for (i in seq_len(steps)) {
        curr = pos[[i]]
        # sample! homoskedastic Gaussian noise from response surface
        on_samp[i] = rnorm(1, resp_on(curr, n), error)
        off_samp[i] = rnorm(1, resp_off(curr, n), error)

        # walk! RWMH for marginal move_pref
        if (i == steps) next()
        proposal = curr + sample(-1L:1L, dim, replace=TRUE)
        while (min(proposal) < 1 || max(proposal) > n) {
            proposal = curr + sample(-1L:1L, dim, replace=TRUE)
        }
        if (runif(1) <= move_pref(proposal[1], n) / move_pref(curr[1], n))
            pos[[i+1]] = proposal
        else
            pos[[i+1]] = curr
    }

    tibble(on = on_samp, off = off_samp,
           pos = pos, idx = map_int(pos, grid_to_index, n=n, dim=dim))
}

# apply sampling policy to a slice of data `x` (output from sample_worm)
# starting from sample `n_start`, collect `n_samp` samples according to `policy`
# `policy` is a grid-shaped array with entries containint the desired
# probability of sampling 'on' at that location.
apply_policy = function(x, n_start, n_samp, policy) {
    stopifnot(max(diff(dim(policy))) == 0)
    range = n_start:(n_start+n_samp-1)
    probs = policy[do.call(rbind, x$pos[range])]
    slice(x, range) %>%
        mutate(which = rbinom(n_samp, 1, probs),
               obs = if_else(which == 1, on, off))
}
