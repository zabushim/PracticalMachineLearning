
library(caret)
library(ggplot2)
library(lattice)
library(randomForest)
if(!file.exists("./data")) {
    dir.create("./data")
}
if (!file.exists("./data/pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile = "./data/pml-training.csv")
}
if (!file.exists("./data/pml-testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile = "~/data/pml-testing.csv")
    
}
PMLTraining <- read.csv("./data/pml-training.csv")
dim(PMLTraining)
names(PMLTraining)
PMLTesting <- read.csv("./data/pml-testing.csv") 
DescColumns <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                 "cvtd_timestamp", "new_window", "num_window")
ColumnsNA <- apply(PMLTraining,2,function(x) table(is.na(x))[1]!=dim(PMLTraining)[1])   
NAColumns <- names(PMLTraining)[ColumnsNA]
DelColumns <- c(DescColumns, NAColumns)
NonNADescPMLTraining <- PMLTraining[, !names(PMLTraining) %in% DelColumns]
ColumnsNZV <- nearZeroVar(NonNADescPMLTraining, saveMetrics = TRUE)
PMLTraining_Ready <- NonNADescPMLTraining[,!ColumnsNZV$nzv]
if(any(is.na(PMLTraining_Ready))==FALSE) {
    "No more NAs"
} else {
    "NAs still present"
}

str(PMLTraining_Ready)
names(PMLTraining_Ready)
ReadyColumns <- names(PMLTraining_Ready)[-53]
PMLTesting <- PMLTesting[,ReadyColumns]
set.seed(201903) 
inTrain <- createDataPartition(PMLTraining_Ready$classe, p = 0.70, list = FALSE)
validationSet <- PMLTraining_Ready[-inTrain, ]
trainingSet <- PMLTraining_Ready[inTrain, ]
TRNCTRL <- trainControl(method = "cv", number = 5, returnData = F)

# To ensure reproducibility we set the seed.
set.seed(201903)
RFmodel <- train(classe ~ ., data = trainingSet, method = "rf", tuneLength = 4, ntree = 100, trControl = TRNCTRL)
#RFmodel <- train(classe ~ ., method="rf", data = trainingSet, tuneLength  = 15, trControl = TRNCTRL) # Random Forest
set.seed(201903)
LDAmodel <- train(classe ~ ., data = trainingSet, method="lda", tuneLength = 4, trControl = TRNCTRL) # Linear Discriminant Analysis
set.seed(201903)
GBMmodel <- train(classe ~ ., data = trainingSet, method="gbm", tuneLength = 4, trControl = TRNCTRL) # Boosted Trees
set.seed(201903)
SVMmodel <- train(classe ~ ., data = trainingSet, method="svmLinear", tuneLength = 4, trControl = TRNCTRL) # Support Vector Machine
resampDist <- resamples(list(RF=RFmodel, LDA=LDAmodel, GBM=GBMmodel, SVM=SVMmodel))
summary(resampDist)
RFose <- 1-mean(resampDist$values[, "RF~Accuracy"])
LDAOse <- 1-mean(resampDist$values[, "LDA~Accuracy"])
GBMose <- 1-mean(resampDist$values[, "GBM~Accuracy"])
SVMose <- 1-mean(resampDist$values[, "SVM~Accuracy"])

dotplot(resampDist)
RFValpred <- predict(RFmodel, newdata=validationSet)
RFValcm <- confusionMatrix(validationSet$classe, RFValpred)
RFValacc <- RFValcm$overall[['Accuracy']]
RFValose <- 1-RFValacc
RFVkappa  <- RFValcm$overall[['Kappa']]
GBMValpred <- predict(GBMmodel, newdata=validationSet)
GBMValcm <- confusionMatrix(validationSet$classe, GBMValpred)
GBMValacc <- GBMValcm$overall[['Accuracy']]
GBMValose <- 1-GBMValacc
GBMVkappa  <- GBMValcm$overall[['Kappa']]
predict(RFmodel, PMLTesting)