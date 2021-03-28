library(reticulate)
library(tidyverse)
library(pacman)
pacman::p_load(devtools,RcppArmadillo)
library(BART)
#setwd("Research/R_RamanathanLab/RLWorms/02_08_Trees/")
#setwd("Research/20_09_07_NewPC/RLWorms/03_24_fixImgs/")
source("./utils.R")


# PARAMETERS
episodes = 10
start_ep = 0
num_save = 500
num_tree = 100
nskip = 200
# Fixed params
n = 12
n_dim = 2
imp_dims = 1:n_dim
error_scale = 5 # rough scale of observation error (sd for specific value)

# Start by formatting for R.
np <- import("numpy")
folder = "./Data/03_28_0/" #######################
fbase = paste(folder,"traj",sep="")

# Making test data
other_locs = rep(floor(n/2), n_dim - length(imp_dims))
d_pred = crossing(pos = map(1:(n^length(imp_dims)), index_to_grid, n, n_dim),
                  which = 0:1) %>%
  filter(map_lgl(pos, ~ all(.[-imp_dims] == other_locs))) %>%
  unnest_wider(pos, names_sep="_")
m_pred = model.matrix(~ 0 + which + ., data=d_pred)


######################################################################
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
  
  
  # Making tree and saving
  fitx <-  wbart(m_bart, d_bart$obs, x.test=m_pred,
            sparse=T, a=0.5*(1+1/length(imp_dims)), nskip=nskip,
            ntree=num_tree, usequants=T, ndpost=num_save, sigquant=0.9,
            nkeeptrain=0)
  np$save(paste(folder,"bart",ind,".npy",sep=""),fitx$yhat.test) # nx288
  np$save(paste(folder,"bartsig",ind,".npy",sep=""),fitx$sigma[-c(1:nskip)]) # n samples
}
