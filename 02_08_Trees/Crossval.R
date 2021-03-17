# Hyperparameter search for BART and DBART
as <- seq(.5,1,.25)
pwrs <- c(1:3)
sigquants <- seq(.1,.9,.4)


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
start_ep = 5
num_save = 500
num_tree = 100
num_train_set = 4000

# Start by formatting for R.
np <- import("numpy")
folder = "./Data/03_08_0/" #######################
fbase = paste(folder,"traj",sep="")
fname <- paste(fbase,ind,".npy",sep="")

# Load file
d <- np$load(fname,allow_pickle=T)
d_bart <- as_tibble(matrix(unlist(d),ncol=4))
d_bart <- d_bart %>% rename(pos_1=V1, pos_2=V2, obs=V3, which=V4)
# Shuffle data
set.seed(42)
rows <- sample(nrow(d_bart))
d_bart <- d_bart[rows,]

# Format for BART
bart_form = obs ~ 0 + which + . 
m_bart = model.matrix(bart_form, data=d_bart)

d_train = m_bart[c(1:num_train_set),]
d_train_obs = d_bart$obs[c(1:num_train_set)]
d_test = m_bart[-c(1:num_train_set),]
d_test_obs = d_bart$obs[-c(1:num_train_set)]

# For each worm episode, wait until episode trajectory file exists.
# In the form of numpy array
mus <- list()
sigs <- list()

for (a in as) {
  for (pwr in pwrs) {
    for (sq in sigquants) {
      print(paste(rho,a,pwr,sq))
      # Making tree and saving
      fit <-  wbart(m_bart, d_bart$obs, x.test=m_pred,
                     sparse=T, a=a, nskip=200,
                     ntree=num_tree, usequants=T, ndpost=num_save,
                     power=pwr, sigquant=sq, nkeeptrain=0, 
                     nkeeptest=100, augment=F, cont=T)
      fit$yhat.test
    }
  }
}
