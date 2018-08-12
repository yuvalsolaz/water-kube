
# coding: utf-8

# # $$CatBoost\ Tutorial$$

# In this tutorial we would explore some base cases of using catboost, such as model training, cross-validation and predicting, as well as some useful features like early stopping,  snapshot support, feature importances and parameters tuning.

# ## $$Contents$$
# * [1. Data Preparation](#$$1.\-Data\-Preparation$$)
#     * [1.1 Data Loading](#1.1-Data-Loading)
#     * [1.2 Feature Preparation](#1.2-Feature-Preparation)
#     * [1.3 Data Splitting](#1.3-Data-Splitting)
# * [2. CatBoost Basics](#$$2.\-CatBoost\-Basics$$)
#     * [2.1 Model Training](#2.1-Model-Training)
#     * [2.2 Model Cross-Validation](#2.2-Model-Cross-Validation)
#     * [2.3 Model Applying](#2.3-Model-Applying)
# * [3. CatBoost Features](#$$3.\-CatBoost\-Features$$)
#     * [3.1 Using the best model](#3.1-Using-the-best-model)
#     * [3.2 Early Stopping](#3.2-Early-Stopping)
#     * [3.3 Using Baseline](#3.3-Using-Baseline)
#     * [3.4 Snapshot Support](#3.4-Snapshot-Support)
#     * [3.5 User Defined Objective Function](#3.5-User-Defined-Objective-Function)
#     * [3.6 User Defined Metric Function](#3.6-User-Defined-Metric-Function)
#     * [3.7 Staged Predict](#3.7-Staged-Predict)
#     * [3.8 Feature Importances](#3.8-Feature-Importances)
#     * [3.9 Eval Metrics](#3.9-Eval-Metrics)
#     * [3.10 Learning Processes Comparison](#3.10-Learning-Processes-Comparison)
#     * [3.11 Model Saving](#3.11-Model-Saving)
# * [4. Parameters Tuning](#$$4.\-Parameters\-Tuning$$)

# ## $$1.\ Data\ Preparation$$
# ### 1.1 Data Loading
# The data for this tutorial can be obtained from [this page](https://www.kaggle.com/c/titanic/data) (you would have to register a kaggle account or just login with facebook or google+)

# In[3]:


from catboost.datasets import titanic
import numpy as np

train_df, test_df = titanic()

train_df.head()


# ### 1.2 Feature Preparation
# First of all let's check how many absent values do we have:

# In[4]:


null_value_stats = train_df.isnull().sum(axis=0)
null_value_stats[null_value_stats != 0]


# As we cat see, **`Age`**, **`Cabin`** and **`Embarked`** indeed have some missing values, so let's fill them with some number way out of their distributions - so the model would be able to easily distinguish between them and take it into account:

# In[5]:


train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)


# Now let's separate features and label variable:

# In[6]:


X = train_df.drop('Survived', axis=1)
y = train_df.Survived


# Pay attention that our features are of differnt types - some of them are numeric, some are categorical, and some are even just strings, which normally should be handled in some specific way (for example encoded with bag-of-words representation). But in our case we could treat these string features just as categorical one - all the heavy lifting is done inside CatBoost. How cool is that? :)

# In[7]:


print(X.dtypes)

categorical_features_indices = np.where(X.dtypes != np.float)[0]


# ### 1.3 Data Splitting
# Let's split the train data into training and validation sets.

# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)

X_test = test_df


# ## $$2.\ CatBoost\ Basics$$
# If you have not already installed ***CatBoost***, you can do so by running the following cell (pre-uncomment it). 

# In[9]:


# ! pip install catboost --use


# Also if you want to draw plots of training you should install **`ipywidgets`** package and run special command before launching jupyter notebook:
# ```
# !pip install ipywidgets
# $ jupyter nbextension enable --py widgetsnbextension
# ```
# We recommend doing this for more convenience.

# Let's make necessary imports.

# In[10]:


from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# ### 2.1 Model Training
# Now let's create the model itself: We would go here with default parameters (as they provide a _really_ good baseline almost all the time), the only thing We would like to specify here is `custom_loss` parameter, as this would give us an ability to see what's going on in terms of this competition metric - accuracy, as well as to be able to watch for logloss, as it would be more smooth on dataset of such size.

# In[11]:


model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42,
    logging_level='Silent'
)


# In[12]:


model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
#     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
);


# As you can see, it is possible to watch our model learn through verbose output or with nice plots (personally I would definately go with the second option - just check out those plots: you can, for example, zoom in areas of interest!)
# 
# With this we can see that the best accuracy value of **0.8341** (on validation set) was acheived on **503th** boosting step.

# ### 2.2 Model Cross-Validation
# 
# It is good to validate your model, but to cross-validate it - even better. And also with plots! So with no more words:

# In[13]:


cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    model.get_params(),
    plot=True
)


# Now we have values of our loss functions at each boosting step averaged by 10 folds, which should provide us with a more accurate estimation of our model performance:

# In[14]:


print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']),
    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],
    np.argmax(cv_data['test-Accuracy-mean'])
))


# In[15]:


print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))


# As we can see, our initial estimation of performance on single validation fold was too optimistic - that is why cross-validation is so important!

# ### 2.3 Model Applying
# All you have to do to get predictions is

# In[16]:


predictions = model.predict(X_test)
predictions_probs = model.predict_proba(X_test)
print(predictions[:10])
print(predictions_probs[:10])


# But let's try to get a better predictions and Catboost features help us in it.

# ## $$3.\ CatBoost\ Features$$
# You may have noticed that on model creation step I've specified not only `custom_loss` but also `random_seed` parameter. That was done in order to make this notebook reproducible - by default catboost chooses some random value for seed:

# In[17]:


model_without_seed = CatBoostClassifier(iterations=10, logging_level='Silent')
model_without_seed.fit(X, y, cat_features=categorical_features_indices)

print('Random seed assigned for this model: {}'.format(model_without_seed.random_seed_))


# Let's define some params and create `Pool` for more convenience. It stores all information about dataset (features, labeles, categorical features indices, weights and and much more).

# In[18]:


params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': False
}
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)


# ### 3.1 Using the best model
# If you essentially have a validation set, it's always better to use the `use_best_model` parameter during training. By default, this parameter is enabled. If it is enabled, the resulting trees ensemble is shrinking to the best iteration.

# In[19]:


model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validate_pool)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model = CatBoostClassifier(**best_model_params)
best_model.fit(train_pool, eval_set=validate_pool);

print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_validation, model.predict(X_validation))
))
print('')

print('Best model validation accuracy: {:.4}'.format(
    accuracy_score(y_validation, best_model.predict(X_validation))
))


# ### 3.2 Early Stopping
# If you essentially have a validation set, it's always easier and better to use early stopping. This feature is similar to the previous one, but only in addition to improving the quality it still saves time.

# In[20]:


# get_ipython().run_cell_magic('time', '', 'model = CatBoostClassifier(**params)\nmodel.fit(train_pool, eval_set=validate_pool)')
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validate_pool)


# In[21]:


earlystop_params = params.copy()
earlystop_params.update({'od_type': 'Iter', 'od_wait': 40})
earlystop_model = CatBoostClassifier(**earlystop_params)
earlystop_model.fit(train_pool, eval_set=validate_pool)

# In[22]:


print('Simple model tree count: {}'.format(model.tree_count_))
print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_validation, model.predict(X_validation))
))
print('')

print('Early-stopped model tree count: {}'.format(earlystop_model.tree_count_))
print('Early-stopped model validation accuracy: {:.4}'.format(
    accuracy_score(y_validation, earlystop_model.predict(X_validation))
))


# So we get better quality in a shorter time.
# 
# Though as was shown earlier simple validation scheme does not precisely describes model out-of-train score (may be biased because of dataset split) it is still nice to track model improvement dynamics - and thereby as we can see from this example it is really good to stop boosting process earlier (before the overfitting kicks in)

# ### 3.3 Using Baseline
# It is posible to use pre-training results (baseline) for training.

# In[23]:


current_params = params.copy()
current_params.update({
    'iterations': 10
})
model = CatBoostClassifier(**current_params).fit(X_train, y_train, categorical_features_indices)
# Get baseline (only with prediction_type='RawFormulaVal')
baseline = model.predict(X_train, prediction_type='RawFormulaVal')
# Fit new model
model.fit(X_train, y_train, categorical_features_indices, baseline=baseline);


# ### 3.4 Snapshot Support
# Catboost supports snapshots. You can use it for recovering training after an interruption or for starting training with previous results. 

# In[24]:


params_with_snapshot = params.copy()
params_with_snapshot.update({
    'iterations': 5,
    'learning_rate': 0.5,
    'save_snapshot': True,
    'logging_level': 'Verbose'
})
model = CatBoostClassifier(**params_with_snapshot).fit(train_pool, eval_set=validate_pool);
params_with_snapshot.update({
    'iterations': 10,
    'learning_rate': 0.1,
})
model = CatBoostClassifier(**params_with_snapshot).fit(train_pool, eval_set=validate_pool);


# ### 3.5 User Defined Objective Function
# It is possible to create your own objective function. Let's create logloss objective function.

# In[25]:


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.
        #
        # In our case, logloss is defined by the following formula:
        # target * log(sigmoid(approx)) + (1 - target) * (1 - sigmoid(approx))
        # where sigmoid(x) = 1 / (1 + e^(-x)).
        
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        
        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = (1 - p) if targets[index] > 0.0 else -p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


# In[26]:


model = CatBoostClassifier(
    iterations=10,
    random_seed=42, 
    loss_function=LoglossObjective(), 
    eval_metric="Logloss"
)
# Fit model
model.fit(train_pool)
# Only prediction_type='RawFormulaVal' is allowed with custom `loss_function`
preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')


# ### 3.6 User Defined Metric Function
# Also it is possible to create your own metric function. Let's create logloss metric function.

# In[27]:


class LoglossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += -w * (target[i] * approx[i] - np.log(1 + np.exp(approx[i])))

        return error_sum, weight_sum


# In[28]:


model = CatBoostClassifier(
    iterations=10,
    random_seed=42, 
    loss_function="Logloss",
    eval_metric=LoglossMetric()
)
# Fit model
model.fit(train_pool)
# Only prediction_type='RawFormulaVal' is allowed with custom `loss_function`
preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')


# ### 3.7 Staged Predict
# CatBoost model has `staged_predict` method. It allows you to iteratively get predictions for a given range of trees.

# In[29]:


model = CatBoostClassifier(iterations=10, random_seed=42, logging_level='Silent').fit(train_pool)
ntree_start, ntree_end, eval_period = 3, 9, 2
predictions_iterator = model.staged_predict(validate_pool, 'Probability', ntree_start, ntree_end, eval_period)
for preds, tree_count in zip(predictions_iterator, range(ntree_start, ntree_end, eval_period)):
    print('First class probabilities using the first {} trees: {}'.format(tree_count, preds[:5, 1]))


# ### 3.8 Feature Importances
# Sometimes it is very important to understand which feature made the greatest contribution to the final result. To do this, the CatBoost model has a `get_feature_importance` method.

# In[30]:


model = CatBoostClassifier(iterations=50, random_seed=42, logging_level='Silent').fit(train_pool)
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# This shows that features **`Sex`** and **`Pclass`** had the biggest influence on the result.

# ### 3.9 Eval Metrics
# The CatBoost has a `eval_metrics` method that allows to calculate a given metrics on a given dataset. And to draw them of course:)

# In[31]:


model = CatBoostClassifier(iterations=50, random_seed=42, logging_level='Silent').fit(train_pool)
eval_metrics = model.eval_metrics(validate_pool, ['AUC'], plot=True)


# In[32]:


print(eval_metrics['AUC'][:6])


# ### 3.10 Learning Processes Comparison
# You can also compare different models learning process on a single plot.

# In[33]:


model1 = CatBoostClassifier(iterations=10, depth=1, train_dir='model_depth_1/', logging_level='Silent')
model1.fit(train_pool, eval_set=validate_pool)
model2 = CatBoostClassifier(iterations=10, depth=5, train_dir='model_depth_5/', logging_level='Silent')
model2.fit(train_pool, eval_set=validate_pool);


# In[34]:


from catboost import MetricVisualizer
widget = MetricVisualizer(['model_depth_1', 'model_depth_5'])
widget.start()


# ### 3.11 Model Saving
# It is always really handy to be able to dump your model to disk (especially if training took some time).

# In[35]:


model = CatBoostClassifier(iterations=10, random_seed=42, logging_level='Silent').fit(train_pool)
model.save_model('catboost_model.dump')
model = CatBoostClassifier()
model.load_model('catboost_model.dump');


# # $$4.\ Parameters\ Tuning$$
# While you could always select optimal number of iterations (boosting steps) by cross-validation and learning curve plots, it is also important to play with some of model parameters, and we would like to pay some special attention to `l2_leaf_reg` and `learning_rate`.
# 
# In this section, we'll select these parameters using the **`hyperopt`** package.

# In[39]:


import hyperopt

def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        learning_rate=params['learning_rate'],
        iterations=500,
        eval_metric='Accuracy',
        random_seed=42,
        logging_level='Silent'
    )
    
    cv_data = cv(
        Pool(X, y, cat_features=categorical_features_indices),
        model.get_params()
    )
    best_accuracy = np.max(cv_data['test-Accuracy-mean'])
    
    return 1 - best_accuracy # as hyperopt minimises


# In[37]:


# !pygmentize ~/anaconda3/lib/python3.6/site-packages/hyperopt/fmin.py


# In[40]:


params_space = {
    'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=50,
    trials=trials
    # rseed=123
)

print(best)


# Now let's get all cv data with best parameters:

# In[41]:


print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))


# Recall that with default parameters out cv score was 0.8283, and thereby we have (probably not statistically significant) some improvement.

# ### Make submission
# Now we would re-train our tuned model on all train data that we have

# In[42]:


model.fit(X, y, cat_features=categorical_features_indices);


# And finally let's prepare the submission file:

# In[ ]:


submisstion = pd.DataFrame()
submisstion['PassengerId'] = X_test['PassengerId']
submisstion['Survived'] = model.predict(X_test)


# In[ ]:


submisstion.to_csv('submission.csv', index=False)

return submisstion.shape


# Finally you can make submission at [Titanic Kaggle competition](https://www.kaggle.com/c/titanic).
# 
# That's it! Now you can play around with CatBoost and win some competitions! :)
