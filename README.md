# Claims-Severity
#installing the libraries required
#install.packages(c("randomForest", "timeit", "caret", "e1071", "mlbench"))
#Loading the libraries to perform randomForest, NaiveBayes, and try xgboost Model
# library(randomForest)
# library(timeit)
# library(caret)
# library(e1071)
# library(data.table)
# library(Matrix)
# library(xgboost)
# library(Metrics)
#Read and store the train dataset into a variable to process it, showProgress displays the progress of the data
RawDataset_For_Training = read.csv('train.csv',na.strings=c(""),header=TRUE)
#Read and store the test dataset into a variable to test the model
RawDataset_For_Testing = read.csv('test.csv',na.strings=c(""),header=TRUE)
#Finding number of rows in dataset and partitioning the data by 30%.
first_30Percent_data=RawDataset_For_Training[1:round((nrow(RawDataset_For_Training))/3),]
last_30Percent_data=RawDataset_For_Training[round(2*((nrow(RawDataset_For_Training))/3)):(nrow(RawDataset_For_Training)-1),]
processRandomForest <-function(x){
r_train=first_30Percent_data[x,2:132]
r_test=last_30Percent_data[x,2:132]
testModel=r_test[,2:129]
r_trainLoss=r_train$loss
r_testLoss=r_test$loss
data_for_randomForestModel=r_train[,54:106]
data_for_randomForestModel$loss=r_trainLoss
r_Model=randomForest(loss ~.,data_for_randomForestModel,importance=TRUE,ntree=100)
r_predictedLoss=predict(r_Model,testModel,type='response')
Accuracy <-round(100-abs(((sum(r_predictedLoss)-sum(r_testLoss))/sum(r_testLoss))*100),2)
a<- "RandomForest Data Accuracy :"
b<- paste(a,Accuracy)
c<- paste(b,"%",sep="")
print(c)
}
cat("##################### Running RandomForest ######################",'\n\n\n')
randomForest_ExecutionTime<- system.time(processRandomForest(c(1:1000)))
cat("RandomForest execution time to process 1000 records and predict the loss : ",randomForest_ExecutionTime["user.self"],'seconds','\n')
randomForest_ExecutionTime<- system.time(processRandomForest(c(1:2000)))
cat("RandomForest execution time to process 2000 records and predict the loss : ",randomForest_ExecutionTime["user.self"],'seconds','\n')
randomForest_ExecutionTime<- system.time(processRandomForest(c(1:3000)))
cat("RandomForest execution time to process 3000 records and predict the loss : ",randomForest_ExecutionTime["user.self"],'seconds','\n')
randomForest_ExecutionTime<- system.time(processRandomForest(c(1:4000)))
cat("RandomForest execution time to process 4000 records and predict the loss : ",randomForest_ExecutionTime["user.self"],'seconds','\n\n')
#Observing the pattern of time to process a,b=a+11,c=b+11*2,d=c+22*2,e=d+44*2
#I've calculated the total time to process 188000
#first 1000 record process time
total_time=3
for(i in 1:round(nrow(RawDataset_For_Training)/1000)){
total_time=total_time+11*i
}
cat("RandomForest total execution time to process 188000 records and predict the loss : ",round(total_time/3600,2),'hours','\n\n')
cat("####################### End RandomForest ########################",'\n\n\n')
cat("###################### Running NaiveBayes #######################",'\n\n\n')
#running NaiveBayes
processNaiveBayes <-function(y){
n_train=first_30Percent_data[y,2:132]
n_test=last_30Percent_data[y,2:132]
n_testDataset=n_test[,2:129]
n_trainLoss=n_train$loss
n_testLoss=n_test$loss
n_model <- naiveBayes(loss ~ ., n_train)
n_predict <- predict(n_model, n_testDataset, type = "raw")
}
naiveBayes_ExecutionTime<- system.time(processNaiveBayes(c(1:100)))
cat("NaiveBayes execution time to process 100 records and predict the loss : ",naiveBayes_ExecutionTime["user.self"],'seconds','\n')
naiveBayes_ExecutionTime<- system.time(processNaiveBayes(c(1:200)))
cat("NaiveBayes execution time to process 200 records and predict the loss : ",naiveBayes_ExecutionTime["user.self"],'seconds','\n')
naiveBayes_ExecutionTime<- system.time(processNaiveBayes(c(1:300)))
cat("NaiveBayes execution time to process 300 records and predict the loss : ",naiveBayes_ExecutionTime["user.self"],'seconds','\n')
naiveBayes_ExecutionTime<- system.time(processNaiveBayes(c(1:400)))
cat("NaiveBayes execution time to process 400 records and predict the loss : ",naiveBayes_ExecutionTime["user.self"],'seconds','\n\n')
cat("NaiveBayes total execution time to process 188000 records and predict the loss :
",round(((nrow(RawDataset_For_Training)/400)*naiveBayes_ExecutionTime["user.self"])/3600,2),'hours','\n\n')
cat("####################### End NaiveBayes ########################",'\n\n\n')
cat("###################### Running XGBoost #######################",'\n\n\n')
#running XGBoost
xg_train = fread("train.csv", showProgress = TRUE)
xg_test = fread("test.csv", showProgress = TRUE)
xg_3_train = log(xg_train[,'loss', with = FALSE] + 200)[['loss']]
xg_train[, c('id', 'loss') := NULL]
xg_test[, c('id') := NULL]
processXgBoostModel <- function(z){
#Creating combined dataset
n_xg_train=nrow(xg_train)
xg_testDataset_trainAndTest=rbind(xg_train, xg_test)
#Fetching header names
headerNames = names(xg_train)
for (test in headerNames) {
if (class(xg_testDataset_trainAndTest[[test]])=="character") {
levels <- sort(unique(xg_testDataset_trainAndTest[[test]]))
xg_testDataset_trainAndTest[[test]] <- as.integer(factor(xg_testDataset_trainAndTest[[test]], levels=levels))
}
}
new_xg_train = xg_testDataset_trainAndTest[1:n_xg_train,c(1:13,16,19,23,25:29,31,36:45,49:54,57,61,64,66,71:73,75:79,80:84,86:130)]
new_xg_test =
xg_testDataset_trainAndTest[(n_xg_train+1):nrow(xg_testDataset_trainAndTest),c(1:13,16,19,23,25:29,31,36:45,49:54,57,61,64,66,71:73,75:79,80:84,8
6:130)]
data_train = xgb.DMatrix(as.matrix(new_xg_train[z]), label=xg_3_train[z])
data_test = xgb.DMatrix(as.matrix(new_xg_test))
parameters_xgboost = list(
seed = 0,
colsample_bytree = 0.5,
subsample = 0.8,
eta = 0.05,
objective = 'reg:linear',
max_depth = 12,
alpha = 1,
gamma = 2,
min_child_weight = 1,
base_score = 7.76
)
best_nrounds = 545
xg_dataset = xgb.train(parameters_xgboost, data_train, nrounds=as.integer(best_nrounds/0.8))
outPut = round(exp(predict(xg_dataset,data_test)) - 200,2)
finalOutput=xg_test
finalOutput$loss=outPut
write.csv(finalOutput,"finalOutput.csv")
}
xgBoost_ExecutionTime<- system.time(processXgBoostModel(c(1:nrow(xg_train))))
cat("XGBoost total execution time to process 188318 records and predict the loss : ",round(xgBoost_ExecutionTime["user.self"]/60,2),'minutes','\n\n')
cat(' *************************************************************************','\n')
cat('Loss for the test dataset is successfully predicted and written into finalOutput.csv file...','\n')
cat('Please check in working directory!','\n\n')
cat("######################### End XGBoost ##########################")
