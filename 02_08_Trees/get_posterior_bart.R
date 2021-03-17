library(reticulate)
library(tidyverse)
library(pacman)
pacman::p_load(devtools,RcppArmadillo)
library(BART)
#setwd("Research/R_RamanathanLab/20_09_07_Repo/RLWorms/02_08_Trees/")
#setwd("Research/20_09_07_NewPC/RLWorms/02_08_Trees/")
source("./worm-sampling/worm-sampling/utils.R")


# PARAMETERS
episodes = 1
start_ep = 29
num_save = 500
num_tree = 50
# Fixed params
n = 12
n_dim = 2
imp_dims = 1:n_dim
error_scale = 20 # rough scale of observation error (sd for specific value)
dim_regularization = 3 # higher values -> more smoothness & sparsity
#num_train_set = 15000

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
  
  #d_train = m_bart[c(1:num_train_set),]
  #d_train_obs = d_bart$obs[c(1:num_train_set)]
  
  # Making test data
  other_locs = rep(floor(n/2), n_dim - length(imp_dims))
  d_pred = crossing(pos = map(1:(n^length(imp_dims)), index_to_grid, n, n_dim),
                    which = 0:1) %>%
    filter(map_lgl(pos, ~ all(.[-imp_dims] == other_locs))) %>%
    unnest_wider(pos, names_sep="_")
  m_pred = model.matrix(~ 0 + which + ., data=d_pred)
  
  # Making tree and saving
  fitx <-  wbart(m_bart, d_bart$obs, x.test=m_pred,
            sparse=T, a=0.5*(1+1/length(imp_dims)), nskip=200,
            ntree=num_tree, usequants=T, ndpost=num_save,
            power=dim_regularization, sigest=error_scale, sigquant=0.1,
            nkeeptest=500)
  np$save(paste(folder,"dbart_50trees",ind,".npy",sep=""),fitx$yhat.test)
}
