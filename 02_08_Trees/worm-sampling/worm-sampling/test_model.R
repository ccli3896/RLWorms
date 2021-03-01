# Main file: test approach on simulated data

# load packages and install if need be
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(BART)) install.packages("BART")

library(tidyverse)
library(BART)
source("dgp.R")
source("utils.R")
source("bart_model.R")

# SET THESE PARAMETERS
n = 12
n_dim = 2
n_batch = 9000
error_scale = 1 # rough scale of observation error (sd for specific value)
dim_regularization = 3 # higher values -> more smoothness & sparsity
thomps_cutoff = 0.1 # reward for sampling 'on' when true surface is within this value of 0

# simulate all of our data
x = sample_worm(n, n_dim, steps=9000)
step = 1
# naive initial policy
policy = array(0.5, dim=rep(n, n_dim))
# our actual collected data
d = tibble()
# dimensions using
imp_dims = 1:n_dim

for (i in 1:20) {
    # simulate our policy
    d = bind_rows(d, mutate(apply_policy(x, step, n_batch, policy), batch=i))
    step = step + n_batch

    res = fit_bart(d, n, n_dim, imp_dims, error_scale, dim_regularization,
                    thomps_cutoff=thomps_cutoff)
    imp_dims = res$imp_dims

    # mean
    plot_d = filter(d, batch==i) %>%
        unnest_wider(pos, names_sep="_")
    p1 = ggplot(res$d_out, aes(pos_1, pos_2, fill=q50)) +
        geom_raster() +
        geom_jitter(aes(fill=NULL, shape=as.integer(3-2*which)),
                    size=0.8, data=plot_d) +
        scale_shape_identity() +
        scale_fill_gradient2() +
        labs(x=NULL, y=NULL, fill="Posterior\nmedian")
    # sd
    p2 = ggplot(res$d_out, aes(pos_1, pos_2, fill=IQR)) +
        geom_raster() +
        geom_jitter(aes(fill=NULL, shape=as.integer(3-2*which)),
                    size=0.8, data=plot_d) +
        scale_shape_identity() +
        labs(x=NULL, y=NULL, fill="Posterior\nIQR")
    # new policy
    p3 = ggplot(res$d_out, aes(pos_1, pos_2, fill=prob)) +
        geom_raster() +
        geom_jitter(aes(fill=NULL, shape=as.integer(3-2*which)),
                    size=0.8, data=plot_d) +
        scale_shape_identity() +
        labs(x=NULL, y=NULL, fill="New\npolicy")
    # 'on' sampling rate
    p4 = group_by(d, batch) %>%
        summarize(pct_on = mean(which == 1)) %>%
    ggplot(aes(batch, pct_on, group=1)) +
        geom_line() +
        geom_point() +
        labs(y="Pct. 'on'")
    gridExtra::grid.arrange(p1, p2, p3, p4, nrow=2, ncol=2, top=paste("Batch", i))

    policy[as.matrix(res$d_out[,imp_dims])] = res$d_out$prob
}
