##########   Bird Bones and Living Habits   ############

setwd('C:/Users/JY/Documents/GitHub/Kaggle/BirdBones')

## load libraries used for this project
library(knitr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(grid)
library(scatterplot3d)
library(e1071)
library(caret)

######----------------------------------- ######
###              Data Exploration            ###
#####-------------------------------------######

# load the data
bird.data <- read.csv("bird.csv")
# check the dimension
n.row <- nrow(bird.data)
n.col <- ncol(bird.data)
print (paste0("# of predictors: ", n.col))
print (paste0("# of observations: ", n.row))
# show how the data look like
head(bird.data)
# remove samples with missing values (7 in 420)
data.complete <- bird.data[rowSums(is.na(bird.data)) == 0, ]


###Check the balance between classes
# compute count for each class
data.count <- melt(table(data.complete[, n.col]))
# class label
label <- c("Scansorial", "Raptors", "Singing", "Swimming", "Terrestrial", "Wading")
# commpute percentage
percent <- rev(data.count$value/sum(data.count$value))
# compute label position
position <- cumsum(rev(data.count$value)) - 0.5*rev(data.count$value)
# check the number of birds in each class
ggplot(data=data.count, aes(x=factor(1), y=value, fill = Var1)) +
  geom_bar(width = 1, stat = "identity")+
  coord_polar(theta = "y") +
  geom_text(aes(label = sprintf("%1.2f%%", 100*percent), y = position)) +
  scale_fill_brewer(palette = "Set2", name = c("Living Habit"),
                    label = label) +
  ggtitle("Figure 1. Proportion of Birds with Different Living Habits") +
  theme(plot.title = element_text(hjust = 0.5, size = 14))


###Check how each bone measure is related to the ecological classes
# function to plot a measure by class
bone.plot <- function(data, var){
  p <- ggplot(data, mapping = aes_string(x = "type", y = var)) +
    geom_boxplot() +
    ggtitle(var) +
    scale_x_discrete(labels=c("Scansorial", "Raptors", "Singing", 
                              "Swimming", "Terrestrial", "Wading")) +
    ggtitle(var)+
    theme(plot.title = element_text(size = 14))
  return (p)
}
# plot for each bone measure
plots <- list()
for (i in 1:(n.col - 2)){
  name <- names(data.complete)[i+1]
  plots[[i]] <- bone.plot(data.complete, name)
}
do.call(grid.arrange, c(plots, ncol=2))


###Correlation between predictors
#correlation between predictors
cor.matrix <- cor(data.complete[, 2:(n.col-1)])
# plot heatmap
melted_cormat <- melt(cor.matrix)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+
  scale_fill_continuous(low = "azure", high = "steelblue", 
                        name = "Correlation") +
  ggtitle("Figure 3. Correlation Between the Bone Measures") +
  theme(plot.title = element_text(hjust = 0.5, size = 14))


######----------------------------------- ######
###   Principal Component Analysis (PCA)     ###
#####-------------------------------------######

# perform PCA
pca <- prcomp(data.complete[2:(n.col-1)], scale=T)

###Variance explained by each principal component (PC)
# check the variance explained by each PC
# total variance of the data
total.var <- sum(pca$sdev^2)
# compute %variance explained by each PC
percent.var <- pca$sdev^2/total.var
# cumulative variance explained
cum.percent <- c()
for (i in 1:(n.col-2)){
  cum.percent <- c(cum.percent, sum(percent.var[1:i]))
}
# plot the accumulative variance against # of PCs
df <- data.frame(percent = cum.percent, 
                 num.PC = seq(1, n.col-2, 1))
ggplot(df, aes(x = num.PC, y = percent))+
  geom_point()+
  geom_line()+
  ggtitle("Figure 4. Cumulative Variance Explained by Principal Components")+
  theme(plot.title = element_text(hjust = 0.5, size = 13))+
  xlab("Number of Principal Components")+
  ylab("Percentage of Variance Explained")


###How the principal components represent the bone measures
score <- data.frame(pca$rotation)
score$names <- rownames(score)
score.df <- melt(score, value.name = "score")

ggplot(score.df, aes(x = names, y = score))+
  geom_bar(stat = "identity")+
  coord_flip()+
  facet_wrap(~ variable)+
  ggtitle("Figure 5. Bone Measures Represented by Each Principal Component")+
  theme(plot.title = element_text(hjust = 0.5, size = 14))+
  ylab("Bone measures")


###Visualize the ecological classes
# combine the PCs with class label into a df
pca.df <- data.frame(pca$x)
pca.df$type <- data.complete$type
# plot and color by sample type
p1<- ggplot(pca.df, aes(x = PC1, y = PC2, color = type)) +
  geom_point() +
  scale_color_discrete(name = "Ecological Class", labels = label)+
  ggtitle("6a. Ecological Classes Shown by PC1 and PC2")+
  theme(plot.title = element_text(size = 13)) 
p2<- ggplot(pca.df, aes(x = PC3, y = PC4, color = type)) +
  geom_point() +
  scale_color_discrete(name = "Ecological Class", labels = label)+
  ggtitle("6b. Ecological Classes Shown by PC3 and PC4")+
  theme(plot.title = element_text(size = 13))
grid.arrange(p1, p2, ncol=2, 
             top = textGrob("Figure 6. Ecological Classes Shown by Principal Components", 
                            gp=gpar(fontsize=16, fontface = 'bold')))


###try binary classes (water vs non-water)
# assign new types to the data
pca.df$new.type <- "Other"
pca.df$new.type[pca.df$type == "SW" | pca.df$type == "W"] <- "Water"
pca.df$new.type <- as.factor(pca.df$new.type)
# 3-D plot to visualize the two groups
colors = c("red", "blue")
colors <- colors[as.numeric(pca.df$new.type)]
plot <- scatterplot3d(pca.df[c(1, 3:4)], 
                      color = colors, angle = 60, pch = 1,
                      main = "Figure 7. Water v.s Other Birds by the PC 2-4")
legend(plot$xyz.convert(-15, 3, 2), legend = levels(pca.df$new.type),
       col = c("red", "blue"), pch =1)


######----------------------------------- ######
###              Classification              ###
#####-------------------------------------######

###Train-test split and PCA transformation
# add the new type to original data
data.complete$new.type <- pca.df$new.type
# number of samples
n.sample <- nrow(data.complete)
# separate training and testing sets
set.seed(322)
index <- sample(seq(1, n.sample, 1), n.sample)
# use 30% as testing set
n.split <- floor(n.sample*0.7)
train.split <- data.complete[index[1:n.split], ]
test.split <- data.complete[index[(n.split+1):n.sample], ]
# apply PCA to the training data
pca.2 <- prcomp(train.split[,2:(n.col-1)], scale = TRUE)
train <- data.frame(pca.2$x)
train[c("type", "new.type")] <- train.split[c("type", "new.type")]
# project the test data to the same PCA scale
test <- data.frame(predict(pca.2, newdata = test.split[, 2:(n.col-1)]))
test[c("type", "new.type")] <- test.split[c("type", "new.type")]


###
##### Water v.s. Non-water birds
###

# check the balance
table(pca.df$new.type)

###Logistic regression and feature selection

# function to store model results
store <- function(model, variable, null.deviance, null.df){
  # compute p-value from Chi-square distribution
  p.val <- 1- pchisq(null.deviance - deviance(model), 
                     length(model$coefficients) - null.df)
  # organize the result into a dataframe
  result <- data.frame(model = variable, 
                       deviance = deviance(model), 
                       num.coef = length(model$coefficients),
                       p.value = p.val)
  return (result)
}

# fit an intercept model
model.0 <- glm(new.type ~ 1, data = train, family = binomial)
result <- store(model.0, "1", deviance(model.0), 1)
# compare to model with 1 predictor
variables <- names(pca.df)[1:10]
for (name in variables){
  formula = as.formula(paste0("new.type ~", name))
  model <- glm(formula, data = train, family = binomial)
  result <- rbind(result, store(model, paste0("1+", name), deviance(model.0), 1))
}
# check the comparison
result

##PC3 is added, next feature
result <- result[c(1, 4), ]
variables <- variables[variables != "PC3"]
# compare to model with 1 more predictor
for (name in variables){
  formula = as.formula(paste0("new.type ~ PC3+", name))
  model <- glm(formula, data = train, family = binomial)
  result <- rbind(result, store(model, paste0("1+PC3+", name), 
                                result$deviance[2], 1))
}
# check the comparison
result

##PC1 is added, next feature
result <- result[c(1:3), ]
variables <- variables[variables != "PC1"]
# compare to model with 1 more predictor
for (name in variables){
  formula = as.formula(paste0("new.type ~ PC3+PC1+", name))
  model <- glm(formula, data = train, family = binomial)
  result <- rbind(result, store(model, paste0("1+PC3+PC1+", name), 
                                result$deviance[3], 1))
}
# check the comparison
result

##PC4 is added
result <- result[c(1:3, 5), ]
variables <- variables[variables != "PC4"]
# compare to model with 1 more predictor
for (name in variables){
  formula = as.formula(paste0("new.type ~ PC3+PC1+PC4+", name))
  model <- glm(formula, data = pca.df, family = binomial)
  result <- rbind(result, store(model, paste0("1+PC3+PC1+PC4+", name), 
                                result$deviance[4], 1))
}
# check the comparison
result

##final model
# check performance of the final model 
model.log <- glm(new.type ~ PC1+PC3+PC4, data = train,
                 family = binomial)
# predict on test set
pred <- predict(model.log, newdata = test)
# convert to binary result
y.pred <- rep("Other", nrow(test))
y.pred[pred > 0.5] <- "Water"
# classification accuracy
print (paste0("Classification accuracy of the logistic model is ", 
              round(mean(y.pred == test$new.type), 4)))
# check confusion matrix
table <- confusionMatrix(y.pred, test$new.type)
table$table


###Support Vector Machines (SVM)
##linear kernel
# perform SVM with linear kernel 
# (the best cost is around the default 1)
svm.linear <- svm(new.type ~ PC1+PC3+PC4, 
                  data = train, kernel = "linear") 
# predict on test set
pred <- predict(svm.linear, newdata = test)
# classification accuracy
print (paste0("Classification accuracy of linear SVM is ",
              round(mean(pred == test$new.type), 4)))

##RBF kernel
set.seed(322)
# perform SVM with RBF kernel using default 10-fold CV
svm.rbf <- tune(svm, new.type ~ PC1+PC3+PC4, 
                data = train, kernel = "radial", 
                ranges = list(gamma = seq(0.01, 0.2, 0.02),
                              cost = seq(1, 200, 20)))
svm.rbf$best.parameters
plot(svm.rbf, main = "Figure 8. Visualize the Grid Search Result")
# predict on test set
pred <- predict(svm.rbf$best.model, newdata = test)
# classification accuracy
print (paste0("Classification accuracy of the radial SVM is ",
              round(mean(pred == test$new.type), 4)))


###
##### Multiclass classification
###

### SVM
##linear kernel
# linear SVM on the multi-class task
log.2 <- tune(svm, type ~ ., data = train, kernel = "linear",
              ranges = list(cost = 10^seq(-3, 3, 1)))
# predict on test set
pred <- predict(log.2$best.model, newdata = test)
# classification accuracy
print (paste0("Multi-class classification accuracy of the Linear SVM is ",
              round(mean(pred == test$type), 4)))
# confusion matrix
table.2 <- confusionMatrix(pred, test$type)
table.2$table

##RBF kernel
# RBF SVM on the multi-class task
rbf.2 <- tune(svm, type ~ ., data = train, kernel = "radial",
              ranges = list(cost = seq(1, 40, 4),
                            gamma = c(seq(0.001, 0.009, 0.002), 
                                      seq(0.01, 1, 0.02))))
rbf.2$best.parameters
# predict on test set
pred <- predict(rbf.2$best.model, newdata = test)
# classification accuracy
print (paste0("Multi-class classification accuracy of the RBF SVM is ",
              round(mean(pred == test$type), 4)))
# confusion matrix
table.3 <- confusionMatrix(pred, test$type)
table.3$table


######----------------------------------- ######
###               Clustering                 ###
#####-------------------------------------######

###
##### Hierachical clustering on bone measures
###

# apply hierachical clustering on bone measures
clusters.bones <- hclust(dist(t(data.complete[2:11])))
# plot the dendrogram
plot(clusters.bones, 
     main = "Figure 9. Hierarchical clustering of the bone measures", 
     xlab = "Bone measures", sub="")
# draw rectangles around clusters
rect.hclust(clusters.bones, h = 800, which = c(1, 2, 3),
            border = c("green", "red", "blue"))


###
##### k-means clustering on living habits
###

# compute % variance explained for different k values
var.explained <- c()
for (k in 1:15){
  model <- kmeans(data.complete[2:11], centers = k)
  var.explained <- c(var.explained, 
                     model$betweenss/(model$tot.withinss + model$betweenss))
}
#plot vairance explained against k values
df <- data.frame(k = 1:15, var.explained = var.explained)
ggplot(df, aes(x = k, y = var.explained))+
  geom_point()+
  geom_line()+
  xlab("Number of Clusters")+
  ylab("Percent Variance Explained")+
  ggtitle("Figure 10. Percent Variance Explained by Different Numbers of Clusters")+
  theme(plot.title = element_text(hjust = 0.5, size = 12))

# perform k-means clustering with 9 clusters
num.clust <- 9
kmeans_4 <- kmeans(data.complete[2:11], centers = num.clust)
# combine results with actual type
type.df <- data.frame(actual = data.complete$type,
                      class = kmeans_4$cluster)
# split by cluster
type.split <- split(type.df, type.df$class)
# classify the cluster according to the majority type
types <- sapply(type.split, 
                function(x) {c = table(x$actual); names(c)[c == max(c)]})
# organize classified type into the data.frame
num_in_clust <- sapply(type.split, nrow)
clust.type <- c()
for (i in 1:num.clust){
  clust.type <- c(clust.type, rep(types[i], num_in_clust[i]))
}
type.df$kmeans.type <- clust.type
# plot stacked bar chart
ggplot(type.df, aes(x = class, fill = actual))+
  geom_bar(stat = "count")+
  scale_fill_brewer(palette = "Set2", name = c("Actual type"),
                    label = label) +
  ggtitle("Figure 11. Actual ecological classes in each cluster") +
  theme(plot.title = element_text(hjust = 0.5, size = 14))+
  scale_x_continuous("Classified type", breaks = 1:num.clust, labels = types)