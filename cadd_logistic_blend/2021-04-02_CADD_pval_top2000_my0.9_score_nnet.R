
library(tidyverse, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(randomForest, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(caret, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")


setwd("/s/cgm/Cejda/_thesis/cadd_pval")


cadd_top2000_train = read.csv("cadd_pval_top_2000_scores_train.csv", header = T)
cadd_2000 = read.table("cadd_pval_top2000.tsv", sep = "\t", header = T)
cadd_2000$SNP_ALT = paste0("X", cadd_2000$SNP, "_", cadd_2000$ALT)
cadd_2000$SNP_ALT = str_replace(cadd_2000$SNP_ALT, ":", ".")

cor_matrix = cor(cadd_top2000_train, method = "pearson")



find_corr = function(cor_matrix, cutoff) {
  if(dim(cor_matrix)[1] != dim(cor_matrix)[2]) {
    stop("Input was not a square matrix")
  }
  col_pairs = list()
  for(i in seq(1:(length(row.names(cor_matrix))-1))) { #Don't go to last row b/c we don't compare diagonal.
    for(j in c((i+1):length(colnames(cor_matrix)))) { # Start at row i + 1 b/c we don't compare diagonal.
      if(cor_matrix[i,j] >= cutoff) {
        temp_list = list(c(row.names(cor_matrix)[i], colnames(cor_matrix)[j])) #create a list of the pair of columns
        col_pairs = append(col_pairs, temp_list) #append this list to the previous list.
      }
    }
  }
  return(col_pairs)
}


pairs = find_corr(cor_matrix, 0.9)

columns_to_remove = list()
for (i in c(1:length(pairs))) {
  score1 = cadd_2000[str_detect(pairs[[i]][1], cadd_2000$SNP_ALT),6]
  score2 = cadd_2000[str_detect(pairs[[i]][2], cadd_2000$SNP_ALT),6]
  
  if (score1 <= score2) {
    columns_to_remove = append(columns_to_remove,pairs[[i]][1])
  }
  else {
    columns_to_remove = append(columns_to_remove, pairs[[i]][2])
  }
}
columns_to_remove = columns_to_remove[!duplicated(columns_to_remove)]
columns_to_remove = unlist(columns_to_remove)

{
  if (length(columns_to_remove) > 0) {
    simplified_top2000_train = cadd_top2000_train[,-which(names(cadd_top2000_train) %in% columns_to_remove)]
  }
  else {
    simplified_top2000_train = cadd_top2000_train
  }
}




cadd_top2000_test = read.csv("cadd_pval_top_2000_scores_test.csv", header = T)


{
  if (length(columns_to_remove) > 0) {
    simplified_top2000_test = cadd_top2000_test[,-which(names(cadd_top2000_test) %in% columns_to_remove)]
  }
  else {
    simplified_top2000_test = cadd_top2000_test
  }
}




write.table(simplified_top2000_train, "simplified_caddpval_top2000_0.9_SCORES_train.csv", row.names = F, quote = F, sep = ",")
write.table(simplified_top2000_test, "simplified_caddpval_top2000_0.9_SCORES_test.csv", row.names = F, quote = F, sep = ",")


set.seed(200)

train = simplified_top2000_train
x_train = model.matrix(PHENOTYPE ~ ., train)[,c(-1)]
y_train = as.factor(train[, 2])

test = simplified_top2000_test
x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])



hyper_grid = expand.grid(
  size = seq(from = 1, to = 5, by = 2),
  decay = c(0,0.01,0.1,0.2,0.5)
)

ctrl <- trainControl(method = "repeatedcv",   # 10 fold cross validation with 3x repeats
                     repeats = 3)

tuned_nnet <- train(
  x = x_train,
  y = y_train,
  method = "nnet",
  metric = "Kappa",
  trControl = ctrl,
  tuneGrid = hyper_grid,
  MaxNWts = 50000
  
)

saveRDS(tuned_nnet, "./tuned_nnet_cadd_pval_top2000_mycorr0.9_score.rds")