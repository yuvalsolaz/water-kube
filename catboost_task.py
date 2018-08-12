
# # $$CatBoost\ Tutorial$$

# In this tutorial we would explore some base cases of using catboost,
# such as model training, cross-validation and predicting, as well as some useful features like early stopping,
# snapshot support, feature importances and parameters tuning.

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



from catboost.datasets import titanic
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
import hyperopt

from sklearn.metrics import accuracy_score

import numpy as np

from htask import HTask

class catboostTask(HTask):

    # hkube functions :

    def on_initialize(self, *args ):
        print('catboost task on_init')
        self.init_params = args[0]
        self._data_loading(self)
        self._data_preperation(self)
        self._data_splitting(self)
        self.send_message('initialized',{'command': 'initialized'})


    def on_start(self):
        super()
        print('on_start catboost')

        # run catboost cross validation with input args at self.input_msg
        self._model_training(self)

        self.send_message('done', {'command': 'done', 'data': {'output': 43}})


    def on_stop(self):
        print('stop')


    def on_done(self):
        print('done')


    # catboost internal functions :

    # region 1.1 Data Loading

    def _data_loading(self):
        # The data for this tutorial can be obtained from [this page](https://www.kaggle.com/c/titanic/data)
        self.train_df, self.test_df = titanic()
        print(self.train_df.head())

    # endregion

    # region 1.2 Data Preparation

    def _data_preperation(self):
        # ### 1.2 Feature Preparation
        # First of all let's check how many absent values do we have:
        self.null_value_stats = train_df.isnull().sum(axis=0)
        self.null_value_stats[null_value_stats != 0]

        # As we cat see, **`Age`**, **`Cabin`** and **`Embarked`** indeed have some missing values,
        # so let's fill them with some number way out of their distributions -
        # so the model would be able to easily distinguish between them and take it into account:
        self.train_df.fillna(-999, inplace=True)
        self.test_df.fillna(-999, inplace=True)

        # Now let's separate features and label variable:
        self.X = train_df.drop('Survived', axis=1)
        self.y = train_df.Survived

        # Pay attention that our features are of differnt types - some of them are numeric, some are categorical,
        # and some are even just strings, which normally should be handled in some specific way
        # (for example encoded with bag-of-words representation).
        # But in our case we could treat these string features just as categorical one -
        # all the heavy lifting is done inside CatBoost. How cool is that? :)
        print(self.X.dtypes)
        self.categorical_features_indices = np.where(self.X.dtypes != np.float)[0]

    # endregion

    # region 1.3 Data Splitting

    def _data_splitting(self):
        # Let's split the train data into training and validation sets.

        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X, self.y,
                                                                                            train_size=0.75,
                                                                                            random_state=42)
        self.X_test = self.test_df

    # endregion

    # region 2.1 Model Training
    # create the model itself:
    # We would go here with default parameters (as they provide a _really_ good baseline almost all the time),
    # the only thing We would like to specify here is `custom_loss` parameter,
    # as this would give us an ability to see what's going on in terms of this competition metric - accuracy,
    # as well as to be able to watch for logloss, as it would be more smooth on dataset of such size.

    def _model_training(self):
        self.model = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=42, logging_level='Silent')
        self.model.fit(self.X_train, self.y_train,
                       cat_features=self.categorical_features_indices,
                       eval_set=(self.X_validation, self.y_validation),
                       logging_level='Verbose',  # you can uncomment this for text output
                       plot=False);

    # endregion

    # region 2.2 Model Cross-Validation
    def _model_cross_validation(self):
        self.cv_data = cv(Pool(self.X, self.y, cat_features=self.categorical_features_indices),
                     self.model.get_params(), plot=False)


        # Now we have values of our loss functions at each boosting step averaged by 10 folds,
        # which should provide us with a more accurate estimation of our model performance:
        print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
              np.max(self.cv_data['test-Accuracy-mean']),
                     self.cv_data['test-Accuracy-std'][np.argmax(self.cv_data['test-Accuracy-mean'])],
                                                       np.argmax(self.cv_data['test-Accuracy-mean'])))

        print('Precise validation accuracy score: {}'.format(np.max(self.cv_data['test-Accuracy-mean'])))

    # endregion

    # region 2.3 Model Applying / predictions

    def _model_prediction(self):
        self.predictions = model.predict(self.X_test)
        self.predictions_probs = model.predict_proba(self.X_test)
        print(self.predictions[:10])
        print(self.predictions_probs[:10])

    # endregion

    # region 4.0 CatBoost parameters tuning

    def _hyperopt_objective(params):
        model = CatBoostClassifier(
            l2_leaf_reg=int(params['l2_leaf_reg']),
            learning_rate=params['learning_rate'],
            iterations=500,
            eval_metric='Accuracy',
            random_seed=42,
            logging_level='Silent' )

        self.cv_data = cv(Pool(self.X, self.y, cat_features=self.categorical_features_indices),self.model.get_params())
        self.best_accuracy = np.max(self.cv_data['test-Accuracy-mean'])
        return 1 - best_accuracy  # as hyperopt minimises

    def _model_optimizations(self):
        # While you could always select optimal number of iterations (boosting steps) by cross-validation and learning curve plots,
        # it is also important to play with some of model parameters,
        # and we would like to pay some special attention to `l2_leaf_reg` and `learning_rate`.
        # In this section, we'll select these parameters using the **`hyperopt`** package.
        params_space = {'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
                        'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1), }

        trials = hyperopt.Trials()

        best = hyperopt.fmin(hyperopt_objective, space=params_space, algo=hyperopt.tpe.suggest, max_evals=50,
                             trials=trials # rseed=123)

        print(best)
        print('Precise validation accuracy score: {}'.format(np.max(self.cv_data['test-Accuracy-mean'])))

    # endregion

