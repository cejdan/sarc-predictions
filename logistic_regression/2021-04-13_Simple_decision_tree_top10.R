
library(tidyverse)
library(rpart)
library(caret)
library(rattle)

setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic")


top10_train = read.table("top_10_matrix_add_clean.raw", header = T)
top10_simple = top10_train[,c(5:16)]
cor_matrix = cor(top10_simple, method = "pearson")

remove = findCorrelation(
  cor_matrix,
  cutoff = 0.90,
  verbose = TRUE,
  names = FALSE,
  exact = TRUE
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



write.table(simplified_top10_train, "simplified_top10_0.9_train.csv", row.names = F, quote = F, sep = ",")
write.table(simplified_top10_test, "simplified_top10_0.9_test.csv", row.names = F, quote = F, sep = ",")



colnames(simplified_top10_train)[3] = "X6.32607969_G"



simplified_top10_train$PHENOTYPE = as.factor(simplified_top10_train$PHENOTYPE)
simplified_top10_train$PHENOTYPE = recode_factor(simplified_top10_train$PHENOTYPE, `1` = "control", `2` = "sarc")
            
            
str(simplified_top10_train)


myfit = rpart(PHENOTYPE ~ SEX + X6.32607969_G, 
              data = simplified_top10_train,
              parms=list(split="gini"),
              control = rpart.control(minsplit = 1, 
                                      minbucket = 1, 
                                      cp = 0.001))

fancyRpartPlot(myfit, cex = 0.7)

