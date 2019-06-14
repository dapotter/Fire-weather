# Unsupervised SOM for fire weather prediction
# SOM is trained on 70% of June 1979 data, tested
# on 30% of  June 1979 data
# Confusion matrix is generated
# How to make ROC curve and AUC?
# Unsupervised Self-Organizing Maps
library(kohonen)
library(pROC)
library(caret)
library(e1071)

############ SET SOM PARAMETER: S############ 
# Data:
# /home/dp/Documents/FWP/R/Training/xy_train.csv
data <- read.csv(file.choose(), header = T)

erc_col = 2       # Column containing ERC

grid_x_dim = 15
grid_y_dim = 15
grid_topo = "hexagonal"
rlen = 200       # Iterations
rad = 15          # radius
rlen_str = as.character(rlen)
#############################################

head(data)
head(data[,-erc_col]) # View predictor variables only (no ERC data)
X <- scale(data[,-erc_col]) # Make X variable containing all predictor variables, drop ERC
summary(X)

# SOM
set.seed(222)

g <- somgrid(xdim = grid_x_dim, ydim = grid_y_dim, topo = grid_topo)

map <- som(X,
           grid = g,
           rlen = rlen,
           alpha = c(0.05, 0.001), # Learning rates: SOM.jl default = 0.2)
           radius = rad)            # Radius: SOM.jl r = sqrt((xdim^2 + ydim^2)/2)

# Learning over time:
plot(map, type='changes')
print(map$changes)
tail(map$changes)
write.csv(map$changes, file =paste("/home/dp/Documents/FWP/R/SOM_changes_",rlen_str,"iters.csv",sep=""), row.names=FALSE)
# Contains the node numbers to which the application belongs
# for all applications in data. There are 400 applications in a 20x20 grid
print(map$unit.classif)
# To see the contents of the row of data that falls into a
# particular node (say node 4):
head(data)
print(map$codes) # Codebook: For each  node, the fan size is based on these numbers

plot(map) # Plot map of variable values in each node:
plot(map, type='codes') # Plot codebook (same asplot(map))
plot(map, type='count') # Plot count of samples (rows) in each node:
plot(map, type='mapping') # Same as above
plot(map, type='dist.neighbours') # Plot distance to neighbors: white means greater dist to neighbors


# -------------------------------------------------
# Supervised Self-Organizing Map
# Data Split
set.seed(123)
# ind contains sample of size 2 (e.g. resulting values will be either 1 or 2)
# sample with replacement
# 70% training data, 30% test data
# train is data where all rows = 1, all columns (70% of rows)
# test is data where all rows = 2, all columns (30% of rows)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
train <- data[ind == 1,]
test <- data[ind == 2,]
head(train)
head(test)

# -------------------------------------------------
# Normalization
trainX <- scale(train[,-erc_col]) # Drop ERC
testX <- scale(test[,-erc_col], # Drop ERC
               center = attr(trainX, "scaled:center"),
               scale = attr(trainX, "scaled:scale"))
trainY <- factor(train[,erc_col])
Y <- factor(test[,erc_col])
test[,erc_col] <- 0
testXY <- list(independent = testX, dependent = test[,erc_col])

# -------------------------------------------------
# Classification & Prediction Model
set.seed(222)
map1 <- xyf(trainX,
            classvec2classmat(factor(trainY)),
            grid = somgrid(grid_x_dim, grid_y_dim, grid_topo),
            rlen = rlen,
            radius = rad)
plot(map1)
plot(map1, type="changes")

# Prediction
pred <- predict(map1, newdata = testXY)
pred_table <- table(Predicted = pred$predictions[[2]], Actual = Y)
cm <- confusionMatrix(pred_table)
cm

# Cluster Boundaries
par(mfrow = c(1,2))
plot(map1, 
     type = 'codes',
     main = c("Codes X", "Codes Y"))
map1.hc <- cutree(hclust(dist(map1$codes[[2]])), 2)
add.cluster.boundaries(map1, map1.hc)
par(mfrow = c(1,1))

