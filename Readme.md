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

**False Positive and False Negative:**

Type I(false Positive) and Type II error(false Negative):

**Accuracy Paradox**
    
    Accuracy rate = correct / Total

**CAP CURVE**

---> Cumulative accuracy profile.

compare model between each other and how much add gain you get.

*Good, poor and random model...*

---> ROC =Receiver Operating Characterstics.

**CAP Curve Analysis:**

-> AR = a(R)/a(p)

x<60-70% Poor
70%v<X<80% Good
<90% Very Good
X <60% Rubbish
x=100% (overfitting model and some depeandant variable and dataset need to adjust).

**Clustering:**
is grouping unlabelled data.

supervised--> regression, cleassification

Unsupervised:--> clustering input data and model have to learn by itself.

**K-MEANS CLUSTERING:**

**Elbow Method:**
approach to make decisions about grouping data .

within cluster sum of squares:

WCSS =distance(Pi, Ci)2

first k means clustering run then we make decisions for WCSS.
more cluster we have smaller WCCS comes and vice versa.

visual method .

K-Means++:
Kmeans++ initialization Algortihm:
1. choose first centroid at random among data points.
2. for each of remaining datapoints compute the distance to the nearest out of alrady selected centroids.
3. Choose next centroid among remaining datapoints using weighted random selection -weighted by D2
4. Repeat step2 and 3 until all k centroids have been selected.
5. Proceed with standard k-means clustering.

 **HC Intuition :**
 What HC does for you:

same as Kmeans but different process.
NOTE:
    *Agglomerative & Divisive:*
    Agglomerative :
    1. make each datapoint single point clsuters-> that forms N cluster.
2. take two closest datapoints and make them one cluster >N-1
3. Take two closest clusters and make them one cluster-> Forms N-2 clusters.
4. Repeat step 3 until there is only one cluster.
-->FIN

Also euclidean distance :

--> distance between two clusters:
1. closest pints.
2. furthest points.
3. Average distance.
4. Distance between centroids.

Agglomerative HC:

**HC Intuition:**

-->How Do Dendograms Work?
  dendograms is like memory of HC .
  more furhter they are more dissimilar they are.

  dendogram contain memory of herirarchical clustering algorithm.

  **HC Intuition:**
  --> Using Dendrograms:
  it also need optimal number of clusters.
Heirarchical approach on larget distance.

--> Knowledge Test:


**Association Rule Learning:**
people who bought also bought...

-->ARL-Movie Recommendation...
Potential Rules:
support,confidence ,list:

support(M)=user watchlists containing M1 and M2/user watchlists containing M1.

Apriori-Support:
Apriori-Confidence:

lift = confidence/support

*Apriori-Algorithm:*

1. Set a min support and confidence.
2. Take all the subsets in transactions having higher support than mininmum support.
3. Take all rules of these subsets having higher confidence than minimum confidence.
4. Sort the rules by decreasing lift.

**Association Rule Learning Eclat Intuition:**

*potential rules:*
recommendation->support(M) = user watchlists containing M/user watchlists
optimisation:-> support(I) =transactions containing I/ transactions

Elcat Algorithm:
1. Set a minimum support.
2. Take all the subsets in trasanctions haning higher support than min support.
3. support these subsets by decreaising support.

**MULTI-ARMED BANDIT PROBLEM:**

**Upper Confidence Bound Intuition(UCB):**

we have d arms.
each time a user connects to this web page , that makesa  round.
at each round n, we choose one ad to display to the user.
At each rouund n, ad I gives reward ri(n) E {0,1}:ri (n) =1 if the user clicked on the ad i, O if user did not.
Our goal is to maximize the total reward we get over many rounds.

**Thompson Sampling Algorithm Intuition:**
 where we think mue values wiilbe:
 We are not trying to guess the distributions behind the machines.

 generated our own bandit configuration:

 *New Round:*
            
   **UCB VS THOMPSON SAMPLING:**
-->UCB is deterministic.
no randomness in it .
Requires update at every round in it.
--> Thompson is Probablistic:
Can accomodate delayed feedback.
better emipirical evidence.


**NLP Intuition:**

types of NLP:
classical vs deep learning:
Bag of words:

**Types of NLP:**

NLP and DL.:
Seq 2 Seq: models for natural language processing.
DNLP:

**Classical VS Deep Learning Model:**

some examples:
1. if else rules (chatbot)
2. Audio frequency components analysis(speech recognition).
3. Bag of words model(classsification).
4. CNN for text recogntion(classification).
5. Seq2Seq(many applications).

  **Bag-of-words:**
--> Yes/No:
--20000 elements long.

[SOS,EOS and Special Words.]

Training data by using principles 

**What is Deep Learning**
Neurons.:

input layer--> Hidden Layer --> Output Layer

interconnect everything with each other...

**plan of Attack:**
The neuron.
The activation function.
how do neural networks work?(example).

How do neural networks learn?
Gradient Descent.
Stochastic Gradient descent.
Back propagation.

**THE NEURON:**

dendrites , axon : 

neuron also have nodes.
neuron gets input signal and gives output signal.

input value pass thorugh synapse for giving value.

mean cancellation --> KL-expansion.

output value canbe :
continous, binary and categorical.
weight are crucial to RNN.

first step and second activation function.
third it passes the signal.

**The activation function:**

Sigmoid function:
rectifier function:

**Hyperbolic Tangent function(tanh):

**HOW DO NN Work?**

**HOW DO NN LEARN:**

input value then actual value 

-->Perceptron:

**Gradient Descent:**

input value --> w1-->activation value -->output value--> actual value -->cost function.

-->curse of dimensionality:

sunway taihulight: world fastest computer.

cross function:

**STOCHASTIC GRADIENT DESCENT:**

cross function need to be convex.
convex into one direction.

batch and stpchastic gradient it run one row at a time.and avoid the problem of local minimum too overall local minimum.
mini batch gradient descent method.

**BACK PROPOGATION:**

forward propogation for input layers.
advance algorithm driven by adjust weight simultaneously.

1. randomly initialize weights to small nmers close to 0.
2. input first observation of dataset input layer ,each future in one input node.
3. forward propogation.
4. compare predicted result to actual result. measure gen error.
5. back prop from left to right error is back prop. update weight according to responsiblity. learning rate decides by how much we update weigths.
6. repeat 1 to 5 and update weights after each observation.
7. when whole training set passed through ANN that makes an epoch. Redo more epochs.

**CONVOLUTIONAL NEURAL NETWORK:**

PLAN OF ATTACK:
-What are CNN?
.Convolution Operation.
--(b)-ReLU Layer.
2. --Pooling.
3. -- Flattening
4. --Full connection
5. Summary
softmax and cross entropy.

**CONVOLUTIONAL NEURAL NETWORK:**

input image--> CNN --> Output label
1. Convolution
2. max Pooling
3. flattening 
4. full connection

**STEP-1:CONVOLUTION**
(f*g)(t) = f(T)g(t-T)dT

input image-->*<--feature detector/filter =Feature Map.
-when feature matches more then feature map get higher value.

create multiple feature map to obtain our first convolution layer.

edge detect: sharpen: emboss: 

**STEP1(B)-ReLU Layer:**

black = negative ;white = positive values 

...Sigmoid--> ReLU-->Leaky ReLU.

**MAX POOLING:**

Our neural network has special invariance .

feature map--> Max pooling ---> Pooled Feature Map.

-->sub sampling: 

input image->convolution->convolution layer->max pooling->pooling layer.

scs.ryerson.ca/

**Flattening:**

pooled feature map --->flattening -->put it in no of rowas and columns.-->input layer of future ANN.

input image--> convolution-->pooling--> flattening-->input layer of ANN.

**FULL CONNECTION:**
-->flattening -->input layer-->fully connected layer--> output layer -->output value.

**SUMMARY**
INPUT IMAGE-->CONVOLUTION-->POOLING-->flattening-->>

**Softmax and Cross-Entropy:**

fj(z) = e^z.j/k.e^z.k 

H(p,q) = -p(x).logq.(x)


when one neural NETwORK OUTPERFORM OTHER THEN CALSSIFICATION ERROR occur.
Mean squared error.

Cross-entropy:

**Principal Component Analysis(PCA)**
noise filtering.
visualisation.
feature extraction.
Stock market predictions.
Gene data analysis.

identify patterns in data.
detect corelation between variables.

standard data.
obtain eigen vectirsa nd values from covariance or corelation matrix.
store it.
construct the projection matrix.

-->e.g: 3D

learn about the relationship betweeen X and Y values.
Find list of principal values/axis.

**Linear Discriminant Analysis:****LDA**

as dimensionality reduction technique.
in preprocessing step fro pattern calssification.
goal to project dataset onto lower-dimensional space.

LDA interested in axes that max the seperation between multiple classes.

goal of LDA to project feature space onto small subspace .

PCA component axes that max the variance.
LDA max component axes the class-seperation.

compute dimnesion mean vectors.
compute scatter metrics.

**K-FOLD CROSS-VALIDATION:**

-->Dataset --> Training set and -->Test Set :

training set split into 10(k)folds.

(1st fold)One fold =V
if aggregate metrics look good, then modeling approach is valid.(train model on full training set)
otherwise adjust hyperparameters or change model and repeat.

=Training data(v) =Validation Data
**(Training on each fold uses same hyper parameters.)**

**BIAS -VARIANCE-TRADEOFF--**
Bias a  systematic error that occurs in machine learning model itself due to incorrect assumptions in ML process.

1. Height bias low variance(model is too simple and doesnt capture the underlying trend of data).
2. Low bias variance( model is too sensitive and capturing noise as if it were real trend(overfitting)).
3. High bias high variance(model too simple to capture the data's trends and too sensitvie capturing noise as well).
4. Low Bias low variacne(Great model... acccurately captures underlying trends of data and genrealizes well to unseen data).

**LOGISTIC REGRESSION INTUITION:**

-->linear regression -->simple and multiple regression:

