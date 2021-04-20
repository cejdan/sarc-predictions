### Neural net visualization

# install.packages("reshape")

library(caret)
library(pROC)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')


setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
rds = readRDS("tuned_nnet_top100_corr0.90.rds")
x_lab = c("SEX",
          "6:32222064_G",
          "6:32438471_C", 
          "6:32441279_A",
          "6:32443258_A", 
          "6:32444056_T", 
          "6:32446112_C",
          "6:32453294_A", 
          "6:32459378_A", 
          "6:32596950_G", 
          "6:32601190_C", 
          "6:32603170_T", 
          "6:32606528_G",
          "6:32607881_G", 
          "6:32607958_G",
          "6:32624899_A", 
          "6:32652509_C") 

# Clean up the labels a little bit:
plot.nnet(rds, cex = 0.8, x.lab = x_lab)  # Looks great!
          
