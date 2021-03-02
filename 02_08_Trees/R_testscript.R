library(reticulate)
library(tidyverse)
library(BART)
setwd("Research/20_09_07_NewPC/RLWorms/02_08_Trees/")
source("./worm-sampling/worm-sampling/utils.R")


# PARAMETERS
n = 12
n_dim = 2
error_scale = 20 # rough scale of observation error (sd for specific value)
dim_regularization = 3 # higher values -> more smoothness & sparsity
thomps_cutoff = 0.1 # reward for sampling 'on' when true surface is within this value of 0
dim_drop = 0.01 # threshold below which we drop dimensions. lower = more conservative
imp_dims = 1:n_dim

# Start by formatting for R.
# Rename columns to Cory's variables
fname = "Data/Rand09-02-12-06/alldatR.npy"
np <- import("numpy")
d <- np$load(fname,allow_pickle=T)
d_bart <- as_tibble(matrix(unlist(d),ncol=4))
d_bart <- d_bart %>% rename(pos_1=V1, pos_2=V2, obs=V3, which=V4)

# Format for BART
bart_form = obs ~ 0 + which + . 
m_bart = model.matrix(bart_form, data=d_bart)

# Fit model (BART with dirichlet prior)
m = wbart(m_bart, d_bart$obs, sparse=T, a=0.5*(1+1/length(imp_dims)),
          ntree=200, usequants=T, ndpost=200,
          power=dim_regularization, sigest=error_scale, sigquant=0.1)

# Dropping dimensions if necessary; making predictions
new_imp_dims = which(m$varprob.mean[-1] >= dim_drop)
other_locs = rep(floor(n/2), n_dim - length(new_imp_dims))
d_pred = crossing(pos = map(1:(n^length(imp_dims)), index_to_grid, n, n_dim),
                  which = 0:1) %>%
  filter(map_lgl(pos, ~ all(.[-new_imp_dims] == other_locs))) %>%
  unnest_wider(pos, names_sep="_")
m_pred = model.matrix(~ 0 + which + ., data=d_pred)

preds = predict(m, m_pred)
preds_diff = preds[,d_pred$which==1] - preds[,d_pred$which==0]

# Save to return to Python
np$save("rcheckdiff200.npy",preds_diff)
np$save("rcheckpreds200.npy",preds)

##################
# SoftBART
##################
library(pacman)
pacman::p_load(devtools,RcppArmadillo)
 #devtools::install_github("theodds/SoftBART", type="source")
library(SoftBart)

new_imp_dims = 1:n_dim
fit <- softbart(X=m_bart, Y=d_bart$obs, X_test=m_pred,
                hypers = Hypers(m_bart, d_bart$obs, num_tree=50, temperature=2),
                opts = Opts(num_burn=300, num_save=500, update_tau=T))
np$save("softbartT2.npy",fit$y_hat_test)
