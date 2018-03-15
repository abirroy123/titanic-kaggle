install.packages("caret")
library(caret)
library(rpart.plot)

TitanicTrain <- read.csv("~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/train.csv", stringsAsFactors = FALSE)
TitanicTest <- read.csv("~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/test.csv", stringsAsFactors = FALSE)

#TitanicTrain$Age[is.na(TitanicTrain$Age)] <- median(TitanicTrain$Age, na.rm=TRUE)
#TitanicTest$Age[is.na(TitanicTest$Age)] <- median(TitanicTest$Age, na.rm=TRUE)
survived <- TitanicTrain$Survived
data.combined <- rbind(TitanicTrain[, -2], TitanicTest)

# Transform some variables to factors
data.combined$Pclass <- as.factor(data.combined$Pclass)
data.combined$Sex <- as.factor(data.combined$Sex)
data.combined$Embarked <- as.factor(data.combined$Embarked)

upper.age <- boxplot.stats(data.combined$Age)$stats[5]
outlier.agefilter <- data.combined$Age < upper.age
age.equation = "Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked"
age.model <- lm(formula = age.equation, data = data.combined[outlier.agefilter,])
age.row <- data.combined[is.na(data.combined$Age),c("Pclass","Sex","SibSp","Parch","Fare" ,"Embarked")]
age.predict <- predict(age.model, newdata = age.row)
data.combined[is.na(data.combined$Age), "Age"] <- age.predict

upper.fare <- boxplot.stats(data.combined$Fare)$stats[5]
outlier.farefilter <- data.combined$Fare < upper.fare
fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(formula = fare.equation, data = data.combined[outlier.farefilter,])
fare.row <- data.combined[is.na(data.combined$Fare),c("Pclass","Sex","Age","SibSp","Parch","Embarked")]
fare.predict <- predict(fare.model, newdata = fare.row)
data.combined[is.na(data.combined$Fare), "Fare"] <- fare.predict

#TitanicTrain[is.na(TitanicTrain$Age), "Age"]
#TitanicTrain[is.na(TitanicTrain$Fare), "Fare"]
#data.combined[is.na(data.combined$Age), "Age"]



str(TitanicTrain)
str(data.combined)

# Split data back out
TitanicTrain <- data.combined[1:891,]
TitanicTrain$Survived <- as.factor(survived)

str(TitanicTrain)

TitanicTest <- data.combined[892:1309,]


# Subset the features we want to use
# features <- c("Survived", "Sex")
features <- c("Survived", "Sex", "Pclass",
              "SibSp", "Parch","Age","Fare","Embarked")

set.seed(12345)

# Use caret to train the rpart decision tree using 10-fold cross 
# validation repeated 3 times and use 15 values for tuning the
# cp parameter for rpart. This code returns the best model trained
# on all the data! Mighty!
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
fitControl2 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")

rpart.cv <- train(Survived~., data=TitanicTrain[,features], method="rpart",metric="Accuracy", tuneLength=15, trControl=fitControl)
rpart.cv


# 891 samples
# 1 predictor
# 2 classes: '0', '1' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 802, 803, 802, 801, 802, 802, ... 
# Resampling results across tuning parameters:
#   
#   cp          Accuracy   Kappa    
# 0.00000000  0.7868213  0.5419915
# 0.03174603  0.7868213  0.5419915
# 0.06349206  0.7868213  0.5419915
# 0.09523810  0.7868213  0.5419915
# 0.12698413  0.7868213  0.5419915
# 0.15873016  0.7868213  0.5419915
# 0.19047619  0.7868213  0.5419915
# 0.22222222  0.7868213  0.5419915
# 0.25396825  0.7868213  0.5419915
# 0.28571429  0.7868213  0.5419915
# 0.31746032  0.7868213  0.5419915
# 0.34920635  0.7868213  0.5419915
# 0.38095238  0.7868213  0.5419915
# 0.41269841  0.7868213  0.5419915  <<<<< Best
# 0.44444444  0.6740853  0.2005122
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was cp = 0.4126984.

rf.cv <- train(Survived~., data=TitanicTrain, method="rf", tuneLength=10, trControl=fitControl,importance = TRUE,ntree = 500)
rf.cv

rf.cv2 <- train(Survived~., data=TitanicTrain[,features], method="rf", tuneLength=7, trControl=fitControl)
rf.cv2

#TitanicTrain
#TitanicTrain[,features]

#mtry <- sqrt(ncol(trainSet))
#tunegrid <- expand.grid(.mtry=sqrt(ncol(TitanicTrain)))
#rf_gridsearch <- train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
#print(rf_gridsearch)


#rf_gridsearch <- train(Survived~., data=TitanicTrain, method="rf",metric="Accuracy", tuneGrid=expand.grid(.mtry=sqrt(ncol(TitanicTrain))), trControl=fitControl2)
#rf_gridsearch

# Random Forest 
# 
# 891 samples
# 7 predictor
# 2 classes: '0', '1' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 802, 802, 801, 803, 801, 802, ... 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 2    0.8283495  0.6170915
# 3    0.8350829  0.6370107
# 4    0.8377006  0.6463704 << THE BEST
# 6    0.8313461  0.6369631
# 7    0.8253617  0.6250487
# 8    0.8231270  0.6208828
# 10    0.8216161  0.6181821
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 4.



cat(paste("\nCross validation standard deviation:",  
          sd(rpart.cv$resample$Accuracy), "\n", sep = " "))

cat(paste("\nCross validation standard deviation:",  
          sd(rf.cv2$resample$Accuracy), "\n", sep = " "))

rpart.best <- rpart.cv$finalModel

rf.best <- rf.cv2$finalModel

rpart.best
rf.best

# Make the model look pretty
#install.packages("rpart.plot")
library(rpart.plot)
prp(rpart.best, type = 0, extra = 1, under = TRUE)


library(randomForest)
# Look at the model - which variable are important?
varImpPlot(rf.best)

fPredTest <- predict(rpart.cv, newdata = TitanicTest, type = "raw")
fPredTestRF <- predict(rf.cv, newdata = TitanicTest, type = "raw")
fPredTestRF2 <- predict(rf.cv, newdata = TitanicTest, type = "raw")
fPredTestRF3 <- predict(rf.cv, newdata = TitanicTest, type = "raw")

fPredTestRF4 <- predict(rf.cv2, newdata = TitanicTest, type = "raw")

# Create dataframe shaped for Kaggle
HWsubmission <- data.frame(PassengerId = TitanicTest$PassengerId,
                         Survived = fPredTest)

HWsubmission_2 <- data.frame(PassengerId = TitanicTest$PassengerId,
                           Survived = fPredTestRF)

HWsubmission_3 <- data.frame(PassengerId = TitanicTest$PassengerId,
                             Survived = fPredTestRF2)

HWsubmission_4 <- data.frame(PassengerId = TitanicTest$PassengerId,
                             Survived = fPredTest)
HWsubmission_5 <- data.frame(PassengerId = TitanicTest$PassengerId,
                             Survived = fPredTestRF3)
HWsubmission_6 <- data.frame(PassengerId = TitanicTest$PassengerId,
                             Survived = fPredTestRF4)



# Write out a .CSV suitable for Kaggle submission
write.csv(HWsubmission_6, file = "~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/MyPredSubmission6.csv", row.names = FALSE)

write.csv(HWsubmission, file = "~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/MyPredSubmission.csv", row.names = FALSE)

write.csv(HWsubmission, file = "~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/MyPredSubmission2.csv", row.names = FALSE)

write.csv(HWsubmission_3, file = "~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/MyPredSubmission3.csv", row.names = FALSE)

write.csv(HWsubmission_4, file = "~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/MyPredSubmission4.csv", row.names = FALSE)

write.csv(HWsubmission_5, file = "~/Documents/DataScienceDojo/bootcamp-master/Kaggle-Capstone/MyPredSubmission5.csv", row.names = FALSE)
