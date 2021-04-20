
library(tidyverse, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(randomForest, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(caret, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(nnet, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
library(kernlab, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")

setwd("/s/cgm/Cejda/_thesis/snplist1_logistic/top100")


top100_train = read.table("top_100_matrix_add_clean.raw", header = T)
top100_simple = top100_train[,c(5:106)]
cor_matrix = cor(top100_simple, method = "pearson")

remove = findCorrelation(
  cor_matrix,
  cutoff = 0.85,
  verbose = TRUE,
  names = FALSE,
  exact = T
)


simplified_top100_train = top100_simple[,-remove]


top100_test = read.table("top_100_test_matrix_add_clean.raw", header = T)
top100_simple_test = top100_test[,c(5:106)]
simplified_top100_test = top100_simple_test[,-remove]


columns_to_fix = simplified_top100_test[,!colnames(simplified_top100_train) %in% colnames(simplified_top100_test), drop = F]
column_names_to_fix = colnames(simplified_top100_test[,!colnames(simplified_top100_train) %in% colnames(simplified_top100_test),drop=FALSE])
correct_column_names = colnames(simplified_top100_train[,!colnames(simplified_top100_train) %in% colnames(simplified_top100_test),drop=FALSE])

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
    names(simplified_top100_test)[names(simplified_top100_test) == column_names_to_fix[i]] = correct_column_names[i]
    simplified_top100_test[,names(simplified_top100_test) == correct_column_names[i]] = columns_to_fix[,i]
  }
}

sum(!colnames(simplified_top100_train) %in% colnames(simplified_top100_test))
#Ok, looks good.



write.table(simplified_top100_train, "simplified_top100_0.85_train.csv", row.names = F, quote = F, sep = ",")
write.table(simplified_top100_test, "simplified_top100_0.85_test.csv", row.names = F, quote = F, sep = ",")


set.seed(200)

train = simplified_top100_train
x_train = model.matrix(PHENOTYPE ~ ., train)[,c(-1)]
y_train = as.factor(train[, 2])

test = simplified_top100_test
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

saveRDS(tuned_svm, "./tuned_SVM_top100_mycorr0.85.rds")