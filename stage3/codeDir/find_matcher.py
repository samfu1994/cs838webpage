import py_entitymatching as em
import os
import copy
import pandas as pd
import csv

sampled_movies = em.read_csv_metadata('datasets/tmp_movies_8.csv', key='id')
sampled_tracks = em.read_csv_metadata('datasets/tmp_tracks_8.csv', key='id')
tbl_labeled = em.read_csv_metadata('datasets/sampled_8.csv', ltable=sampled_movies, rtable=sampled_tracks)

# spliting data into training and testing sets
train_test = em.split_train_test(tbl_labeled, train_proportion=0.7)

dev_set = train_test['train']
eval_set = train_test['test']
em.to_csv_metadata(dev_set, 'datasets/dev_set.csv')
em.to_csv_metadata(eval_set, 'datasets/eval_set.csv')

# myset = em.split_train_test(dev_set, train_proportion=0.9)
# I_set = myset['train']
# J_set = myset['test']
# em.to_csv_metadata(I_set, 'datasets/I_set.csv')
# em.to_csv_metadata(J_set, 'datasets/J_set.csv')

# creating feature for matching
match_t = em.get_tokenizers_for_matching()
match_s = em.get_sim_funs_for_matching()
atypes1 = em.get_attr_types(sampled_movies)
atypes2 = em.get_attr_types(sampled_tracks)
match_c = em.get_attr_corres(sampled_movies, sampled_tracks)
match_f = em.get_features(sampled_movies, sampled_tracks, atypes1, atypes2, match_c, match_t, match_s)

# generating feature vectors
H = em.extract_feature_vecs(dev_set, 
                            feature_table=match_f, 
                            attrs_after='label',
                            show_progress=False)

# filling missing values in feature vectors
H.fillna(value=0, inplace=True)

# creating a set of learning-based matchers
dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
nb = em.NBMatcher(name='NaiveBayes')

# Selecting the best matcher using cross-validation

# precision of matchers for 5-fold cross validations
result_p= em.select_matcher([dt, svm, rf, lg, ln, nb], table=H, 
        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
        k=5,
        target_attr='label', metric='precision', random_state=0)
result_p['cv_stats']

# recall of matchers for 5-fold cross validations
result_r= em.select_matcher([dt, svm, rf, lg, ln, nb], table=H, 
        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
        k=5,
        target_attr='label', metric='recall', random_state=0)
result_r['cv_stats']

# F1 of matchers for 5-fold cross validations
result_f1 = em.select_matcher([dt, svm, rf, lg, ln, nb], table=H, 
        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
        k=5,
        target_attr='label', metric='f1', random_state=0)
result_f1['cv_stats']

# evaluating the selected matcher

# Converting evaluation dataset into feature vectors
L = em.extract_feature_vecs(eval_set, feature_table=match_f,
                            attrs_after='label', show_progress=False)

L.fillna(value=0, inplace=True)

# Training the best matcher using feature vectors from development set
rf.fit(table=H, 
       exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
       target_attr='label')

# predicting over evaluation set
predictions = rf.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
              append=True, target_attr='predicted', inplace=False)

# evaluating the predictions
eval_result = em.eval_matches(predictions, 'label', 'predicted')
em.print_eval_summary(eval_result)

