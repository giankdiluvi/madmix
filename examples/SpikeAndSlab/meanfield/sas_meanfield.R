### ### ### ### ### ### ### ### ### ### ### ### ### #
### ### ### ### ### ### ### ### ### ### ### ### ### #
# Run Mean-Field VI on two Spike-And-Slab examples  #
### ### ### ### ### ### ### ### ### ### ### ### ### #
### ### ### ### ### ### ### ### ### ### ### ### ### #

# preamble ####
library(tidyverse)
library(sparsevb)


### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ###
### ### # PROSTATE CANCER ### ### #
### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ###

# import prst data ####
prst_dat_x = readr::read_csv('../dat/prst_dat_x.csv',
                             col_names = FALSE)
prst_dat_y = readr::read_csv('../dat/prst_dat_y.csv',
                             col_names = FALSE)

# fit prst model ####
prst_model = sparsevb::svb.fit(X = prst_dat_x %>% as.matrix(),
                               Y = prst_dat_y %>% pull(),
                               family = "linear",
                               slab = "gaussian")


# save prst results ####
prst_results = tibble::tibble(
  mu    = prst_model$mu,
  sigma = prst_model$sigma,
  pi    = prst_model$gamma
)
readr::write_csv(prst_results,"../results/mf_prst_results.csv")




### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ###
### ###  SUPERCONDUCTIVITY  ### ###
### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ###

# import spr data ####
spr_dat_x = readr::read_csv('../dat/spr_dat_x.csv',
                             col_names = FALSE)
spr_dat_y = readr::read_csv('../dat/spr_dat_y.csv',
                             col_names = FALSE)

# fit spr model ####
spr_model = sparsevb::svb.fit(X = spr_dat_x %>% as.matrix(),
                              Y = spr_dat_y %>% pull(),
                              family = "linear",
                              slab = "gaussian")


# save spr results ####
spr_results = tibble::tibble(
  mu    = spr_model$mu,
  sigma = spr_model$sigma,
  pi    = spr_model$gamma
)
readr::write_csv(spr_results,"../results/mf_spr_results.csv")
