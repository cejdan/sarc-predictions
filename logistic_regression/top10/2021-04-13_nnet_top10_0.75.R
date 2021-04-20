
library(tidyverse, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(randomForest, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(caret, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(nnet, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(kernlab, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")

setwd("/s/cgm/Cejda/_thesis/snplist1_logistic/top10")


top10_train = read.table("top_10_matrix_add_clean.raw", header = T)
top10_simple = top10_train[,c(5:16)]
cor_matrix = cor(top10_simple, method = "pearson")

remove = findCorrelation(
  cor_matrix,
  cutoff = 0.75,
  verbose = TRUE,
  names = FALSE,
  exact = T
)


simplified_top10_train = top10_simple[,-remove]


top10_test = read.table("top_10_test_matrix_add_clean.raw", header = T)
top10_simple_test = top10_test[,c(5:16)]
simplified_top10_test = top10_simple_test[,-remove]


columns_to_fix = simplified_top10_test[,!colnames(simplified_top10_train) %in% colnames(simplified_top10_test), drop = F]
column_names_to_fix = colnames(simplified_top10_test[,!colnames(simplified_top10_train) %in% colnames(simplified_top10_test),drop=FALSE])
correct_column_names = colnames(simplified_top10_train[,!colnames(simplified_top10_train) %in% colnames(simplified_top10_test),drop=FALSE])

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
    names(simplified_top10_test)[names(simplified_top10_test) == column_names_to_fix[i]] = correct_column_names[i]
    simplified_top10_test[,names(simplified_top10_test) == correct_column_names[i]] = columns_to_fix[,i]
  }
}

sum(!colnames(simplified_top10_train) %in% colnames(simplified_top10_test))
#Ok, looks good.



write.table(simplified_top10_train, "simplified_top10_0.75_train.csv", row.names = F, quote = F, sep = ",")
write.table(simplified_top10_test, "simplified_top10_0.75_test.csv", row.names = F, quote = F, sep = ",")


set.seed(200)

train = simplified_top10_train
x_train = model.matrix(PHENOTYPE ~ ., train)[,c(-1)]
y_train = as.factor(train[, 2])

test = simplified_top10_test
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

saveRDS(tuned_nnet, "./tuned_nnet_top10_corr0.75.rds")