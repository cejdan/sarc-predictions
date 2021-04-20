

library(ggplot2)
library(caret)
library(tidyverse)
library(pROC)

results_df = data.frame(model = NA, feature_selection = NA, sample = NA, rsq_threshold = NA, 
                        AUC = NA, mtry = NA, accuracy = NA, kappa = NA, 
                        accuracy_null = NA, accuracy_pval = NA, sensitivity = NA, specificity = NA
                        )

{
# Random Forest - Logistic - top10 - 0.95
model = "RF"
feature_selection = "logistic"
sample = "top10"
rsq_threshold = 0.95

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top10")
rds = readRDS("tuned_forest_top10_corr0.95_logistic.rds")
test = read.csv("simplified_top10_0.95_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])

ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)
results_df = results_df[2,]
row.names(results_df) = NULL


############
# Top 100


# Random Forest - Logistic - top100 - 0.95
model = "RF"
feature_selection = "logistic"
sample = "top100"
rsq_threshold = 0.95

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
rds = readRDS("tuned_forest_top100_corr0.95_logistic.rds")
test = read.csv("simplified_top100_0.95_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top100 - 0.90
model = "RF"
feature_selection = "logistic"
sample = "top100"
rsq_threshold = 0.90

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
rds = readRDS("tuned_forest_top100_corr0.90_logistic.rds")
test = read.csv("simplified_top100_0.90_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top100 - 0.85
model = "RF"
feature_selection = "logistic"
sample = "top100"
rsq_threshold = 0.85

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
rds = readRDS("tuned_forest_top100_corr0.85_logistic.rds")
test = read.csv("simplified_top100_0.85_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top100 - 0.80
model = "RF"
feature_selection = "logistic"
sample = "top100"
rsq_threshold = 0.80

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
rds = readRDS("tuned_forest_top100_corr0.80_logistic.rds")
test = read.csv("simplified_top100_0.80_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top100 - 0.75
model = "RF"
feature_selection = "logistic"
sample = "top100"
rsq_threshold = 0.75

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
rds = readRDS("tuned_forest_top100_corr0.75_logistic.rds")
test = read.csv("simplified_top100_0.75_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


###########################
# Top 500


# Random Forest - Logistic - top500 - 0.95
model = "RF"
feature_selection = "logistic"
sample = "top500"
rsq_threshold = 0.95

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
rds = readRDS("tuned_forest_top500_corr0.95_logistic.rds")
test = read.csv("simplified_top500_0.95_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top500 - 0.90
model = "RF"
feature_selection = "logistic"
sample = "top500"
rsq_threshold = 0.90

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
rds = readRDS("tuned_forest_top500_corr0.90_logistic.rds")
test = read.csv("simplified_top500_0.90_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top500 - 0.85
model = "RF"
feature_selection = "logistic"
sample = "top500"
rsq_threshold = 0.85

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
rds = readRDS("tuned_forest_top500_corr0.85_logistic.rds")
test = read.csv("simplified_top500_0.85_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top500 - 0.80
model = "RF"
feature_selection = "logistic"
sample = "top500"
rsq_threshold = 0.80

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
rds = readRDS("tuned_forest_top500_corr0.80_logistic.rds")
test = read.csv("simplified_top500_0.80_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)



# Random Forest - Logistic - top500 - 0.75
model = "RF"
feature_selection = "logistic"
sample = "top500"
rsq_threshold = 0.75

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
rds = readRDS("tuned_forest_top500_corr0.75_logistic.rds")
test = read.csv("simplified_top500_0.75_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top1000 - 0.95
model = "RF"
feature_selection = "logistic"
sample = "top1000"
rsq_threshold = 0.95

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
rds = readRDS("tuned_forest_top1000_corr0.95_logistic.rds")
test = read.csv("simplified_top1000_0.95_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top1000 - 0.90
model = "RF"
feature_selection = "logistic"
sample = "top1000"
rsq_threshold = 0.90

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
rds = readRDS("tuned_forest_top1000_corr0.90_logistic.rds")
test = read.csv("simplified_top1000_0.90_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top1000 - 0.85
model = "RF"
feature_selection = "logistic"
sample = "top1000"
rsq_threshold = 0.85

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
rds = readRDS("tuned_forest_top1000_corr0.85_logistic.rds")
test = read.csv("simplified_top1000_0.85_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)



# Random Forest - Logistic - top1000 - 0.80
model = "RF"
feature_selection = "logistic"
sample = "top1000"
rsq_threshold = 0.80

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
rds = readRDS("tuned_forest_top1000_corr0.80_logistic.rds")
test = read.csv("simplified_top1000_0.80_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top1000 - 0.75
model = "RF"
feature_selection = "logistic"
sample = "top1000"
rsq_threshold = 0.75

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
rds = readRDS("tuned_forest_top1000_corr0.75_logistic.rds")
test = read.csv("simplified_top1000_0.75_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - Logistic - top2000 - 0.95
model = "RF"
feature_selection = "logistic"
sample = "top2000"
rsq_threshold = 0.95

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
rds = readRDS("tuned_forest_top2000_corr0.95_logistic.rds")
test = read.csv("simplified_top2000_0.95_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top2000 - 0.90
model = "RF"
feature_selection = "logistic"
sample = "top2000"
rsq_threshold = 0.90

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
rds = readRDS("tuned_forest_top2000_corr0.90_logistic.rds")
test = read.csv("simplified_top2000_0.90_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top2000 - 0.85
model = "RF"
feature_selection = "logistic"
sample = "top2000"
rsq_threshold = 0.85

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
rds = readRDS("tuned_forest_top2000_corr0.85_logistic.rds")
test = read.csv("simplified_top2000_0.85_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top2000 - 0.80
model = "RF"
feature_selection = "logistic"
sample = "top2000"
rsq_threshold = 0.80

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
rds = readRDS("tuned_forest_top2000_corr0.80_logistic.rds")
test = read.csv("simplified_top2000_0.80_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)

# Random Forest - Logistic - top2000 - 0.75
model = "RF"
feature_selection = "logistic"
sample = "top2000"
rsq_threshold = 0.75

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
rds = readRDS("tuned_forest_top2000_corr0.75_logistic.rds")
test = read.csv("simplified_top2000_0.75_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)



# Random Forest - LASSO - 0.95
model = "RF"
feature_selection = "LASSO"
sample = "LASSO"
rsq_threshold = 0.95

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
rds = readRDS("tuned_forest_lasso_mycorr0.95.rds")
test = read.csv("simplified_lasso_0.95_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
levels(y_test) = c("control", "sarc")
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - LASSO - 0.90
model = "RF"
feature_selection = "LASSO"
sample = "LASSO"
rsq_threshold = 0.90

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
rds = readRDS("tuned_forest_lasso_mycorr0.90.rds")
test = read.csv("simplified_lasso_0.90_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
levels(y_test) = c("control", "sarc")
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - LASSO - 0.85
model = "RF"
feature_selection = "LASSO"
sample = "LASSO"
rsq_threshold = 0.85

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
rds = readRDS("tuned_forest_lasso_mycorr0.85.rds")
test = read.csv("simplified_lasso_0.85_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
levels(y_test) = c("control", "sarc")
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)


# Random Forest - LASSO - 0.80
model = "RF"
feature_selection = "LASSO"
sample = "LASSO"
rsq_threshold = 0.80

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
rds = readRDS("tuned_forest_lasso_mycorr0.80.rds")
test = read.csv("simplified_lasso_0.80_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
levels(y_test) = c("control", "sarc")
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)



# Random Forest - LASSO - 0.75
model = "RF"
feature_selection = "LASSO"
sample = "LASSO"
rsq_threshold = 0.75

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
rds = readRDS("tuned_forest_lasso_mycorr0.75.rds")
test = read.csv("simplified_lasso_0.75_test.csv")

x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
y_test = as.factor(test[,2])
levels(y_test) = c("control", "sarc")
pred_probs = predict(rds, x_test, type = "prob")
pred_class = predict(rds, x_test)
cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
accuracy = as.numeric(cmatrix$overall[1])
kappa = as.numeric(cmatrix$overall[2])
accuracy_null = as.numeric(cmatrix$overall[5])
accuracy_pval = as.numeric(cmatrix$overall[6])
sensitivity = as.numeric(cmatrix$byClass[1])
specificity = as.numeric(cmatrix$byClass[2])
ROC = roc(y_test, pred_probs[,2])
AUC = auc(ROC)
mtry = rds$bestTune[,1]
temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)

results_df = rbind(results_df, temp_df)




}



results_df$sample = factor(results_df$sample,levels = c("top10", "top100", "top500", "top1000", "top2000", "LASSO"))

ggplot(data = results_df, aes(x = sample, y=AUC, fill = rsq_threshold)) +
  geom_bar(stat="identity", position=position_dodge2(preserve = "single")) +
  coord_cartesian(ylim = c(0.5,0.61)) +
  ggtitle("Random Forest Results - AUC")


ggplot(data = results_df, aes(x = sample, y=kappa, fill = rsq_threshold)) +
  geom_bar(stat="identity", position=position_dodge2(preserve = "single")) +
  coord_cartesian(ylim = c(0,0.16)) +
  ggtitle("Random Forest Results - Kappa") +
  scale_fill_gradient(low = "green4", high = "lightgreen")



ease_of_use = results_df[,c(3,4,7,10,8,11,12,5)]
write.table(ease_of_use, "easy_rf_results.csv", sep=",", quote=F, row.names = F)
