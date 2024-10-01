**Data preprocessing:**
import the data.
clean the data.
split into training and test sets

Modelling:
build the model
train model
make predictions.

Evaluation:
calculate performance metrics.
make a verdict.

**Splitting the training set and test set:**
predicted and actual values :

**feature scaling**
is always apply to columns.
normalization and standardization:
x'=x-xmin/xmax-xmin for n:and x'=x-mue/sigma  for s:

Section:3**Data preprocessing**


**simple linear regression:**
y =bo+b1x1

**ordinary least squares:**
work in visual sense.
sum(yi -yi^)2 is minimized.

**Multiple linear regression:**
y^ = b0+ b1x1 + b2x2 +--+bnXn

Assumptions of linear regression:
1. linearity( linear relation between Y and each X)
2. Homoscedasticity(equal variance)
3. Multivariate Normality(mormality of error distribution)
4. independance (of observations. includes "no autocorrelation")
5. lack of multicollinearity(predictors are not corelated with each other)
6. the outlier check ()

Dummy variables:
works like light switches.
dependant and independant variables.
y =b0 + b1x1 + b2x2 +b3x3   +b4D1

Dummy var Trap:
y =b0 + b1x1 + b2x2 +b3x3   +b4D1+b5D2
cant have constant and dummy variables at a same time.
Always omit one dummy variable.

D2 =1-D1

**Understanding the P Value:**->(statistics for BA and datascience A-Z)

Statistical Signifinance:
e.g : coin toss (two possible outcomes).
Ho =fair coin
H1 =not a fair coin 

**Multiple linear regression intitution:**
building a model:
 
 x1,x2,x3,x4,x5,x6,x7 -> y
 
 1.in = out
 2.

 **5 methods of building models:**
 -> 1.All-in:

.prior knowledge; OR
.You have to; OP
.preparing for backward elimination

 ->2.Backward elimination->stepwise regression.:

1. select signifinance level to stay in the model.
2. fit the model with all possible predictors.
3. consider the predictor with highest P values . if P> SL goto step 4 toherwise goto FIN.
4. Remove the predictor.
5. Fit model without this variable.

 ->.3.forward selection->stepwise regression.:

 1.select signifinace levelto entrance the model
 2. fit the simple regression models y -xn select the one with lowest P-value.
 3. keep this variable and fit all the possible models with one extra predictor added to the one(s) you already have
 4. consider predictor with lowest P-value if P< SL, goto step3  ,otherwise goto FIN.(keep the previous model)

 ->4.bidirectinal elimination->stepwise regression.:

  1. select a signifinace level to enter and stay in the model
  2. perform next step step of ofrward selection (new variables must have P <SLENTER to enter>)
  3. Perform all steps of backward elimination (old var must have P < SLSTAY to stay>)
  4. No new variables can enteer and no old variables can exit.(model is ready).
 
 5.Score comparison
 All possible models.

 1. select a criterion of goodness of fit
 2. construct all possible regression models 2^N-1 total combinations.
 3. select the one with the best criterion
 model is ready 
 e.g:-> 10 columns mean 1023 models.


**Polynomial Regression**
y =b0+b1x1+ b2x1^2 +---+ bnx1n

**SVR Intuition**
Support vector regression:

Insensitive Tube: 

**Heads-up About non linear SVR:**

SVM intuition.

Section on kernel SVM:

mapping to higher dimension
kernel trick
types of K.F
non linear kernel SVR.

**Decision Tree Intuition**

->classification and regression trees :

**Random Forest Intuition:**

Ensemble learning:
1.Pick at random K datapoints from training set.
2.Build decision tree associated to these K datapoints.
3. Choose the number N tree of trees you want to builld and repeat steps 1 and 2.
4.for new datapoints maek each one of oyur Ntree predict the value of Y to for the datapoint in question and assign new data point the average acroos all of the predicted Y values.

**R Squared**

SSres= SUM(yi -yi^)2
SStot = SUM(yi - yavg)2

R^2 =1 - SSres / SStot

Ruleof thumb -> 1.0 = perfict fit 
0.9 =very good 
<0.7 =not great
<0.4 =terrible 
<0 = model makes no snese for this data

**Adjusted R Squared:**

R2 is between 0 and  1
R2 is goodness of fit (greattor is better)

R2 = 1-SS(res)/SS(tot)
SS(tot) does nto change 
SS(res) will decrease or stay the same(this is because of ordinary least squares: SS(res) -> Min)

y^ = b0 +b1x1+ b2x2 + b3x3
Solution
(Adj R2 = 1-(1-R2) * n-1/n-k-1)


**Classification**
to identify category of new observations based on training data.

**Logistic Regression:**
predict categorical dependant variable from a number of indepedant variables.

y-axis as binary outcomes(0 and 1).
its curve is called sigmoid curve.

     ln p/1-p=b(o) +b1x1.
this will give us probablities.

multiple independant variables.

**Maximum Likelihood:**

likelihod =multiplying number on right side by number on left side numbers.

best curve <= maximum likelihood.

**K - Nearest Neighbours(KNN)**
 divides the data into category 

1. choose the number K of neighbours.
2. Take K nearest neighbours of new data points , according to euclidean distance.
3. Among these K neigbours count the number of datapoints in each category.
4. Assign the new datpoints to the category where you counted the most neighbours.
You model is ready...

Applies on graphs.
1. for first step calcultion :Euclidean distance between P1 and P2  = (x2-x1)2+(y2-y1)2
2. then divide them into categiry 


**SVM Intuition:**


How to seperate these points?

Hyperplanes:

by Maximun Margin:
sum of distance have to be maximumize to draw it

support vectors:

Maximum Margin Hyperplane(maximum margin classifier):

positive hyperplane 
Negative hyperplane

-> What so special about SVM'S:

**KERNEL SVM:**

SVM seperate well these points.
why?
bz data is not linearly seperable.

linearly and non linearly seperable.

**Mapping to higher dimension:**

**A higher dimensional Space:**

mapping function:

fie(x1,x2) = (x1,x2,z)

mapping to higher dimension space canbe highly compute-intensive.

**THE KERNEL TRICK**

The Gaussian RBF Kernel:

K(x,l) = e- ||x-li||/2sigma(2)

for our dataset seperations.
>0 its green
>=0 its red in color.

**Types of kernel functions:**

Gaussian RBF kernel:

Sigmoid kernel:
K(X,Y) = tanh(y.X(T).Y + r)

Polyoial Kernel:
K(X,Y) = (y.X(T)Y + r)d, y>0


**Non Linear SVR(Advanced)**

**Baye's Theorem:**

What's the probablity:

P(A|B)= P(B|A)*P(A)/P(B)

**Naive bayes Classifier Intuition:**
use classes concept too.

Plan of Attack:

likelihood of person:
1. prior probablity
2. marginal likelihood-> P(X)=Number of similar observations/total observations.

3. likelihood-> P(X/Walks)=Among those who walk/total number of walkers

4. posterior probality->

 step:3 comparance of algorithms.


**Naive Bayes Classifier Intuition:**
Addition comments:

1. why naive?
.independance assumption
2. P(X)-> number of similar observations/total observations

3. comparison of both ones.

-> more than two classes ?
then you have to calculate one is enough for solutions.

**Decision Tree Intuition:**

Cart -> Classification trees and regression trees.

Rewind: 
it makes decision step by step on further demand.

old method .
reborn with upgrades.
random forest.
gradient boosting etc..

**Random Forest Classification:**

->*Ensemble learning:*
pick at random K datapoints from training set.
build the decision tree associated tothese K datapoints.

choose the number N trees you want to build and repeat seps 1 and 2 
for a new datapoint, make each one of oyur Ntrees predict the category to which data points belongs and assign the new datapoints to the category that wins the majority vote.


**Confusion Matrix and Accuracy:**
prediction and actual data dependancy.

Accuracy rate and error rate:
AR =correct/total =TN+TP/Total

ER = Incorrect/Total = FP+ FN/total
