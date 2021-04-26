
library(tidyverse, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(randomForest, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(caret, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(nnet, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(kernlab, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")


setwd("/s/cgm/Cejda/_thesis/snplist2_lasso")


lasso = read.table("lasso.lasso", header = T)
lasso$SNP_ALT = paste0("X", lasso$SNP, "_", lasso$A1)
lasso$SNP_ALT = str_replace(lasso$SNP_ALT, ":", ".")

lasso_train = read.table("lasso_matrix_add_clean.raw", header = T)
lasso_simple = lasso_train[,c(5:490)]

cor_matrix = cor(lasso_simple, method = "pearson")


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


pairs = find_corr(cor_matrix, 0.95)

columns_to_remove = list()
for (i in c(1:length(pairs))) {
  score1 = lasso[str_detect(pairs[[i]][1], lasso$SNP_ALT),4]
  score2 = lasso[str_detect(pairs[[i]][2], lasso$SNP_ALT),4]
  
  if (abs(score1) >= abs(score2)) {
    columns_to_remove = append(columns_to_remove,pairs[[i]][1])
  }
  else {
    columns_to_remove = append(columns_to_remove, pairs[[i]][2])
  }
}



columns_to_remove = columns_to_remove[!duplicated(columns_to_remove)]
columns_to_remove = unlist(columns_to_remove)


if (length(columns_to_remove) > 0) {
  lasso_simple = lasso_simple[,-which(names(lasso_simple) %in% columns_to_remove)]
}




lasso_test = read.table("lasso_test_matrix_add_clean.raw", header = T)
lasso_simple_test = lasso_test[,c(5:490)]

if (length(columns_to_remove) > 0) {
  lasso_simple_test = lasso_simple_test[,-which(names(lasso_simple_test) %in% columns_to_remove)]
}




columns_to_fix = lasso_simple_test[,!colnames(lasso_simple) %in% colnames(lasso_simple_test), drop = F]
column_names_to_fix = colnames(lasso_simple_test[,!colnames(lasso_simple) %in% colnames(lasso_simple_test),drop=FALSE])
correct_column_names = colnames(lasso_simple[,!colnames(lasso_simple) %in% colnames(lasso_simple_test),drop=FALSE])

if (length(columns_to_fix) > 0) {
  
  for (i in seq(from = 1, to = length(columns_to_fix), by = 1)) {
    for (j in seq(from = 1, to = length(columns_to_fix[,i]), by = 1)) {
      if (columns_to_fix[,i][j] == 0) {
        columns_to_fix[,i][j] = 2
      }
      else if (columns_to_fix[,i][j] == 2) {
        columns_to_fix[,i][j] = 0
      }
    }
  }
  for (i in seq(from = 1, to = length(columns_to_fix), by = 1)) {
    names(lasso_simple_test)[names(lasso_simple_test) == column_names_to_fix[i]] = correct_column_names[i]
    lasso_simple_test[,names(lasso_simple_test) == correct_column_names[i]] = columns_to_fix[,i]
  }
}

sum(!colnames(lasso_simple) %in% colnames(lasso_simple_test))
#Ok, looks good.



write.table(lasso_simple, "simplified_lasso_0.95_train.csv", row.names = F, quote = F, sep = ",")
write.table(lasso_simple_test, "simplified_lasso_0.95_test.csv", row.names = F, quote = F, sep = ",")


set.seed(200)

train = lasso_simple
x_train = model.matrix(PHENOTYPE ~ ., train)[,c(-1)]
y_train = as.factor(train[, 2])

test = lasso_simple_test
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

saveRDS(tuned_svm, "./tuned_svmRadial_lasso_mycorr0.95.rds")