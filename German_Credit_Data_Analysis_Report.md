**German Credit Data Analysis**

**Introduction:**

Based on the applicant&#39;s profile the German bank has to decide whether to go ahead with the loan approval or not.The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes.

**About the data:**

German data consists of numerical and categorical data. The response variable is categorical with 2 levels- good and bad. The variables in the data are classified as,

**Numerical:** Duration in month, credit amount, Installment rate in percentage of disposable income, Present residence since, Age, Number of people being liable to provide maintenance for,Number of existing credits at this bank.

**Categorical:** Status of existing checking account, Credit history, Credit amount, Present employment since, Personal status and sex, Other debtors / guarantors, Property, Other installment plans, Housing, Job, Telephone, foreign worker.

**Main Goal:**

Whileclassifying customers/applicants into good or bad with classification models, the model which classifies bad customers correctly is preferable. It is worse to class a customer as good when they are bad, than it is to class a customer as bad when they are good. Hence our class of importance in all the models is &quot;Bad&quot;. The sensitivity and accuracy for these models is evaluated in the analysis. The model with high sensitivity that which classifies or predicts most of the bad customers as bad is chosen.

**Classification Algorithms:**

I will use Logistic Regression, Decision Trees and Neural networks to classify and predict the response variable that is to classify either the customer is good or bad.

**Logistic Regression:**

Logistic Regression helps in prediction when the response variable is categorical. Hence our data can be modeled with it.

**Data Preparation:**

The response variable in the data is Y indicating &quot;1&quot; as Good and &quot;2 &quot;as Bad. In logistic model, we get the prediction as a probability since it uses the logit function. Hence the response variable

Y is changed to (0,1) where 1 indicates &quot;Bad&quot; and 0 indicates &quot;Good&quot;.

First all the variables in the data are considered for the logistic fit. I created dummy variables for categorical variables with different levels.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_1.png)

Next, I did rescaling for continuous variables since all variables are in different scale. This helps in maintaining all the variables in same standards and equal likely to the modeling. The sample output would look like below after creating dummies and rescaling. Else the &quot;CRED\_AMT&quot; – credit amount is ranging in 1000s which is way far from other variables.

The model is first trained and then tested on the validation data. Here, 80% random sample is considered as train data and 20% of data is considered as validation data. That means 800 observation into train data and 200 observations into validation data.

**Analysis:**

Logistic model is fit with all the variables in the train data. Next step regression is used for variable selection. Significant variables are chosen from the logistic model.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_2.png) 



![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_3.png)

The summary of step wise regression gives the variables selected from original logistic model with null deviance of 950.61 and residual deviance of 703.59. The corresponding standard error and estimate of each variable selected can

be drawn. The AIC value of the model is 786.91.

From the above model, we could know that for approving loan of an applicant at the German bank the following criteria is considered,

Status of existing checking account (more than 0DM), Duration in month, Credit history,Purpose (except for new car),Credit amount,Savings account/bonds(not less than 100 DM) ,Present employment (has to be employed), Installment rate in percentage of disposable income, Other debtors / guarantors (Should be a guarantor /co-applicant), Age, Number of existing credits at this bank,Should not be foreign worker.

And remaining characteristics of the applicant is not taken into consideration since we focus not to class a customer as good when they are bad than to class a customer as bad when they are good.

The model designed on train data is now tested on validation data to evaluate the performance of the model. With the predicted model in &quot;pred.val&quot;, the cutoff value is calculated for minimum misclassification cost.

 ![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_4.png)

The cost curve shows that, the misclassification cost has decreased as the cutoff value increased up to certain value and then increased. We could see a dip at 0.16 which is now chosen as out cutoff value to classify the applicants after prediction.

From the below confusion matrix,

Our class of importance here is &quot;bad&quot; and the model was able to predict 66 applicants as &quot;Bad&quot; who are actually bad out of 200 applicants. 72 applicants are predicted correctly to be &quot;Good&quot; applicants. We could see high sensitivity in the model which indicates that the model is pretty good (88%) in identifying bad applicants as bad which is our goal of analysis. Though it has low specificity, this model can be considered as better model with the accuracy of predicting bad applicants of 69% percent.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_5.png) 

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_6.png)

Area under the curve: 0.8068

The total accuracy of the model can be known from the above ROC curve. The area under the curve is 0.8068 which us near to 1. Hence the model is a better model to classify applicants of German bank.

**Classification with Decision Trees:**

Decision trees classify the response, or the dependent variable based on the regressor variables. It accepts categorical and continuous independent variables. Decision tree has the in-built feature of selecting the significant variables while fitting the model. So, German bank data can be analyzed with original data.

**Data Preparation:**

Original data is around 1000 observations. It is always better to partition the data so that the performance of the model fitted can be evaluated comparing to original values. The data is now divided into train and validation data. 80% of original data into train and 20% into validation is chosen.

**Analysis:**

Now constructing a classification tree on train data with &quot;rpart&quot; function in R.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_7.png)

The classification tree constructed can be plotted with prp function and the result would be a tree which directs the new observations to classify as a &quot;Good&quot; or &quot;Bad&quot; applicant.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_8.png)

From the above plot we can see that,

We get the prediction of validation data based on the constructed model. The main aim of this analysis is to classify actual &quot;Bad&quot; applicants as &quot;Bad&quot; and it would be worse when classified them as &quot;Good&quot; than it is to class a customer as bad when they are good. Hence our class of importance is &quot;bad&quot; which is represented by &#39;2&#39; in the data.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_10.png)

We can see only 12 bad customers out of 200 who are classified as good. The sensitivity of this prediction is 84% which is a good sign to our model. The specificity of the model is 0.52 and accuracy is 0.64. Though the specificity and accuracy are a bit low, I believe this prediction made on validation data to be significant and this is a better model.

The idea of cost complexity pruning is to identify a tree of optimal size that has the minimum cost complexity. I pruned the model and obtained the below result. From this, we need to select the cost complexity value which is corresponding to the minimum xerror value.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_11.png) 

The table above provides the cross-complexity parameter (CP) and the cross-validation error (xerror) for 24 trees of increasing depth. nsplit denotes the number of splits for the associated tree, rel error reflects the error on the training sample, xstd shows the estimated standard error for the cross validation error. The root node error is 0.71. The minimum value of xerror would be 1.64 and the corresponding nsplit for this xerror is 108 (109 terminal nodes) which may provide the optimum cp value.

This can be verified from the evaluation of performance of this pruned model on validation data.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_12.png) 

The sensitivity of the model is 0.48 which is low compared to original model. This indicates that the bad customers are not correctly predicted as bad.

I tried taking the cp value as 0.173913 which corresponds to the xerror value 2.6939. After this value in the table, the xerror value increased and it decreased all the way down. I will check the performance of this tree.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_13.png)

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_14.png)

From the above tree, I can confirm that, this is the better pruned model since the sensitivity of the model increased to 0.88. The accuracy is 0.65 which is better than previous model. From this model, most of the bad customers are correctly predicted as bad and the count if 9 applicants out of 200. The model eliminated most of the variables and selected Status of existing checking account, employment time period till date, Purpose of application of loan, Properties and the duration.

**Classification with Neural Networks:**

A neural network is trained by adjusting neuron input weights based on the network&#39;s performance on example inputs. If the network classifies an image correctly, weights contributing to the correct answer are increased, while other weights are decreased.

**Data Preparation:**

Neural network accepts continuous regressor variables. German data is a mixed dataset with categorical and continuous variables. Hence dummies are created for all variables in the data which are categorical. The continuous variables are been rescaled so that all the variables are in same standards. I removed the original categorical variables from the data. Th response variable is a numeric datatype, that is been changed to factor and created dummies for response variable as well. Original data is around 1000 observations. It is always better to partition the data so that the performance of the model fitted can be evaluated comparing to original values. The data is now divided into train and validation data. 70% of original data into train and 30% into validation is taken.

**Analysis:**

The train data is taken into neural net function which results in the neural net model. Here Good and Bad columns which are created as dummies are fitted with all other variables in the data. Here we get two outputs with Good and Bad. The sample output is shown. These are the predicted probabilities as good and bad customer. We have to set a cutoff to classify these into both categories. Here &quot;bad&quot; is considered as the class of importance and labeled as 1.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_15.png)

Fig 1 

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_16.png)

Fig 2

Fig1 shows the plot of neural network with 2 layers and 2 nodes. Fig 2 gives the output in terms of probability of good and bad customers.


From the above cost curve, the cut off 0.35 has the minimum misclassification cost. With this cutoff value the prediction is done on validation data and the accuracy of the model is 0.76. The sensitivity of the model is 0.63 which indicates that only 63% of bad customers as correctly predicted in validation sample. But our goal is to reduce the number of bad customers who are predicted as good. For that, the input weights of the neural nets are to be adjusted so that the model accuracy and sensitivity increases.

I am considering the neural network model with 1 layer and 4 nodes. This changes the input weights in the model and the resulting prediction of good and bad customers will change. The model is now fit with these input changes. The output is shown below.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_17.png)

The cost curve of the neural net model shows the cutoff value 0.35 has the minimum misclassification rate. Hence that cutoff is taken and predicted the performance of validation data.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_18.png)

We can see that the sensitivity of the model increased to 0.72 from previous model which means 72%of the bad customers are correctly predicted in the validation data. There are only 28 bad applicants who are predicted wrong as good customers. The accuracy of the model is also increased. Hence this model can be considered as the better significant model with neural net algorithm.

![](https://github.com/ravalibongoni33/German-Credit-Data-Analysis-with-Machine-Learning-Algorithms/blob/master/images/German_Credit/German_19.png)

The area under the ROC curve estimated with predicted model of 4 nodes is, 0.65.

**Conclusion:**

In logistic model, the accuracy of the chosen model is 0.69 with sensitivity 0.88 and AOC is 0.80. The cutoff calculated for this model is 0.16

In decision trees, the accuracy of the chosen model is 0.65 with sensitivity 0.88 and cp value is 0.173913.

In Neural networks, the accuracy of the chosen model is 0.63 with sensitivity 0.72 and AOC is 0.65. The cutoff calculated for this model is 0.35.

Comparing all these classification models, as our main goal is to classify most of the bad customers as bad with class of importance bad, the sensitivity of the model should be high.

Hence, I believe, Logistic regression and Decision trees are better models with 0.88 sensitivity.
