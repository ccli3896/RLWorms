library(reticulate)
library(tidyverse)
library(pacman)
pacman::p_load(devtools,RcppArmadillo)
library(SoftBart)
#setwd("Research/20_09_07_NewPC/RLWorms/02_08_Trees/")
source("./worm-sampling/worm-sampling/utils.R")


# PARAMETERS
episodes = 5
start_ep = 25
num_save = 500
num_tree = 100
temp = 1 # temperature
# Fixed params
n = 12
n_dim = 2
imp_dims = 1:n_dim

# Start by formatting for R.
np <- import("numpy")
folder = "./Data/03_08_0/" #######################
fbase = paste(folder,"traj",sep="")

# For each worm episode, wait until episode trajectory file exists.
# In the form of numpy array
for (ind in c(start_ep:(start_ep+episodes-1))) {
  fname <- paste(fbase,ind,".npy",sep="")
  print(paste("Waiting for",fname))
  while (!file.exists(fname)) {
    Sys.sleep(1)
  }
  print(paste(fname,"appeared"))
  # Load file
  d <- np$load(fname,allow_pickle=T)
  d_bart <- as_tibble(matrix(unlist(d),ncol=4))
  d_bart <- d_bart %>% rename(pos_1=V1, pos_2=V2, obs=V3, which=V4)
  
  # Format for BART
  bart_form = obs ~ 0 + which + . 
  m_bart = model.matrix(bart_form, data=d_bart)
  
  # Making test data
  other_locs = rep(floor(n/2), n_dim - length(imp_dims))
  d_pred = crossing(pos = map(1:(n^length(imp_dims)), index_to_grid, n, n_dim),
                    which = 0:1) %>%
    filter(map_lgl(pos, ~ all(.[-imp_dims] == other_locs))) %>%
    unnest_wider(pos, names_sep="_")
  m_pred = model.matrix(~ 0 + which + ., data=d_pred)
  
  # Making tree and saving
  fit <- softbart(X=m_bart, Y=d_bart$obs, X_test=m_pred,
                  hypers = Hypers(m_bart, d_bart$obs, num_tree=num_tree, temperature=temp),
                  opts = Opts(num_burn=200, num_save=num_save, update_tau=T))
  #np$save(paste(folder,"sbart",ind,".npy",sep=""),fit$y_hat_test)
}
