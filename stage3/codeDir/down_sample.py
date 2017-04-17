#match songs/tracks and movies/tracks
#for songs/tracks , attrs are song_name
#for movies/tracks, attrs are movie_name, year
#songs: title, 
#tracks: title aka movie_name, year, song aka song_name,
#movies: title, year

# _5 : 1500/ 350
# _6 : 1800/ 500
# _7 : 1800/430
# _8 : 1800 / 400
import py_entitymatching as em
import os
import copy
import pandas as pd
import csv

movies = em.read_csv_metadata('datasets/movies.csv', key='id')
tracks = em.read_csv_metadata('datasets/tracks.csv', key='id')

movies['title'] = movies['title'].str.lower()
tracks['title'] = tracks['title'].str.lower()

movies['title'] = movies['title'].str.replace(r"\(.*\)","")
movies['title'] = movies['title'].str.replace(r"\[.*\]","")
movies['title'] = movies['title'].str.replace(r"\*.*\*","")
tracks['title'] = tracks['title'].str.replace(r"\(.*\)","")
tracks['title'] = tracks['title'].str.replace(r"\[.*\]","")
tracks['title'] = tracks['title'].str.replace(r"\*.*\*","")

em.to_csv_metadata(movies, 'datasets/processed_a_8.csv');
em.to_csv_metadata(tracks, 'datasets/processed_b_8.csv');

processed_A = em.read_csv_metadata('datasets/processed_a_8.csv');
processed_B = em.read_csv_metadata('datasets/processed_b_8.csv');

print ("enter down_sample")
sample_movies, sample_tracks = em.down_sample(movies, tracks, size=1800, y_param=2, show_progress=False)
em.set_key(sample_movies, 'id')
em.set_key(sample_tracks, 'id')
em.to_csv_metadata(sample_movies, 'datasets/tmp_movies_8.csv')
em.to_csv_metadata(sample_tracks, 'datasets/tmp_tracks_8.csv')
sample_movies = em.read_csv_metadata('datasets/tmp_movies_8.csv')
sample_tracks = em.read_csv_metadata('datasets/tmp_tracks_8.csv')


print ("enter blocking")
ob = em.OverlapBlocker()
ab = em.AttrEquivalenceBlocker()
rb = em.RuleBasedBlocker()

C1 = ab.block_tables(sample_movies, sample_tracks, 'year', 'year', l_output_attrs=['title', 'year'], r_output_attrs=['title','year'])


C2 = ob.block_candset(C1, 'title', 'title', word_level=True, rem_stop_words=True, overlap_size=1)
# C2 = ob.block_tables(sample_movies, sample_tracks, 'title', 'title', word_level=True, rem_stop_words=True, 
# 					overlap_size=1, l_output_attrs=['title', 'year'], 
#                     r_output_attrs=['title', 'year'],
#                     show_progress=False)

block_f = em.get_features_for_blocking(sample_movies, sample_tracks);

rb.add_rule(['title_title_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) < 0.6', 'title_title_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.6'] ,block_f)
C3 = rb.block_candset(C1, n_jobs=-1,show_progress = False)


# C3 = ab.block_candset(C1, l_block_attr='year', r_block_attr='year')

D = em.combine_blocker_outputs_via_union([C2, C3])

# C3 = C1
# Use block_tables to apply blocking over two input tables.


# corres = [('title', 'title'), ('year', 'year')]
# sample_movies3 = em.read_csv_metadata('datasets/tmp_movies.csv')
# sample_tracks3 = em.read_csv_metadata('datasets/tmp_tracks.csv')
# D = em.debug_blocker(C3, sample_movies3, sample_tracks3, attr_corres=corres)

em.to_csv_metadata(D, 'datasets/tbl_blocked_8.csv');
tbl_blocked = em.read_csv_metadata('datasets/tbl_blocked_8.csv',\
 ltable=sample_movies, rtable=sample_tracks)

S = em.sample_table(tbl_blocked, 400)
em.to_csv_metadata(S, 'datasets/sampled_8.csv')


with open('metadata_8.csv', 'wb') as data1:
	writer = csv.writer(data1, delimiter = ',', quotechar='|')
	for entry in S.values:
		l = len(entry)
		item = entry[-4:]
		writer.writerow(item)

match_f = em.get_features_for_matching(sample_movies, sample_tracks)
H = em.extract_feature_vecs(S, feature_table=match_f)
with open('data_8.csv', 'wb') as data:
	writer = csv.writer(data, delimiter = ',', quotechar='|')
	flag = 0
	names = []
	for idx, row in H.iterrows():
		item = []
		print row
		for it in row.iteritems():
			if flag:
				names.append(it[0])
			item.append(it[1])
		flag = 1
		writer.writerow(item)
	print names