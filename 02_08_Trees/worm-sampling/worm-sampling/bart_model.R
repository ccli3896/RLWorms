# Run BART model
library(BART)


# fit BART model
# d is data
# n, n_dim specify state space
# imp_dims is vector of dimensions to consider
# erorr_scale is rough error scale
# dim_regularization should be higher for more sparsity
# dim_drop is the threshold below which we drop dimensions. lower = more conservative
fit_bart = function(d, n, n_dim, imp_dims=1:n_dim, error_scale=0.5,
                    dim_regularization=3, dim_drop=0.01, thomps_cutoff=0.1) {
    # group to save data
    d_bart = d %>%
        select(pos, which, obs) %>%
        mutate(pos = map(pos, ~ .[imp_dims])) %>%
        unnest_wider(pos, names_sep="_") %>%
        group_by(across(starts_with("pos_")), which) %>%
        summarize(obs = mean(obs),
                  n = as.numeric(n()))

    bart_form = obs ~ 0 + which + . - n
    m_bart = model.matrix(bart_form, data=d_bart)

    cat("Fitting model...\n")
    sink("/dev/null")
    m = wbart(m_bart, d_bart$obs, sparse=TRUE, a=0.5*(1+1/length(imp_dims)),
              ntree=200, usequants=T, ndpost=800,
              power=dim_regularization, sigest=error_scale, sigquant=0.5,
              w=d_bart$n)
    sink()

    cat("Making predictions...\n")
    new_imp_dims = which(m$varprob.mean[-1] >= dim_drop)
    other_locs = rep(floor(n/2), n_dim - length(new_imp_dims))
    d_pred = crossing(pos = map(1:(n^length(imp_dims)), index_to_grid, n, n_dim),
                      which = 0:1) %>%
        filter(map_lgl(pos, ~ all(.[-new_imp_dims] == other_locs))) %>%
        unnest_wider(pos, names_sep="_")
    m_pred = model.matrix(~ 0 + which + ., data=d_pred)

    sink("/dev/null")
    preds = predict(m, m_pred)
    sink()
    preds_diff = preds[,d_pred$which==1] - preds[,d_pred$which==0]

    cat("Updating policy...\n")
    preds = t(apply(preds_diff, 2, quantile, c(0.25, 0.5, 0.75)))
    colnames(preds) = c("q25", "q50", "q75")
    region_policy = apply(preds_diff, 2, function(x) mean(abs(x) <= thomps_cutoff))
    d_out = bind_cols(filter(d_pred, which==1), as_tibble(preds)) %>%
        mutate(policy = region_policy,
               IQR = (q75 - q25),
               prob = exp(-abs(q50) / IQR))

    list(d_out = d_out,
         imp_dims = imp_dims[new_imp_dims])
}

