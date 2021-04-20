
library(tidyverse, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(randomForest, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(caret, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(nnet, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")


setwd("/s/cgm/Cejda/_thesis/fathmm_pval")


fathmm_top500_train = read.csv("fathmm_pval_top_500_scores_train.csv", header = T)
fathmm_500 = read.table("fathmm_pval_top500.tsv", sep = "\t", header = T)
fathmm_500$SNP_ALT = paste0("X", fathmm_500$ID, "_", fathmm_500$variant)
fathmm_500$SNP_ALT = str_replace(fathmm_500$SNP_ALT, ":", ".")

cor_matrix = cor(fathmm_top500_train, method = "pearson")


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
  score1 = fathmm_500[str_detect(pairs[[i]][1], fathmm_500$SNP_ALT),6]
  score2 = fathmm_500[str_detect(pairs[[i]][2], fathmm_500$SNP_ALT),6]
  
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
    simplified_top500_train = fathmm_top500_train[,-which(names(fathmm_top500_train) %in% columns_to_remove)]
  }
  else {
    simplified_top500_train = fathmm_top500_train
  }
}



fathmm_top500_test = read.csv("fathmm_pval_top_500_scores_test.csv", header = T)



{
  if (length(columns_to_remove) > 0) {
    simplified_top500_test = fathmm_top500_test[,-which(names(fathmm_top500_test) %in% columns_to_remove)]
  }
  else {
    simplified_top500_test = fathmm_top500_test
  }
}



write.table(simplified_top500_train, "simplified_fathmm_pval_top500_my0.9_SCORES_train.csv", row.names = F, quote = F, sep = ",")
write.table(simplified_top500_test, "simplified_fathmm_pval_top500_my0.9_SCORES_test.csv", row.names = F, quote = F, sep = ",")


set.seed(500)

train = simplified_top500_train
x_train = model.matrix(PHENOTYPE ~ ., train)[,c(-1)]
y_train = as.factor(train[, 2])

test = simplified_top500_test
x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])


levels(y_train) = c("control", "sarc")

ctrl <- trainControl(method = "repeatedcv",   # 10 fold cross validation with 3x repeats
                     repeats = 3,
                     classProbs =  TRUE)

tuned_svm <- train(
  x = x_train,
  y = y_train,
  method = "svmRadial",
  tuneLength = 15,
  metric = "Kappa",
  trControl = ctrl
)

saveRDS(tuned_svm, "./tuned_svmRadial_fathmm_pval_top500_mycorr0.9_score.rds")