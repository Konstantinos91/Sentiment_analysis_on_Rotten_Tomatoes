# Sentiment_analysis_on_Rotten_Tomatoes
The demonstration utilized as a part of the huge domains of information retrieval and natural language processing is the bag-of-words approach for the sentiment classification analysis of (Rotten Tomatoes dataset) movie reviews. Manifoldness is being maintained by using this method. Nevertheless, to be more specific, a text, for instance, a single sentence or a record is illustrated as the pack which contains its own words, neglecting syntax and even word arrange. Regarding the analysis, every sentence is being parsed into the tree structure of its own, and each hub is appointed an estimation mark going from 0 - 4, where the numbers stands for exceptionally negative, negative, neutral, positive and extremely positive respectively. The corpus comprises of around 150,000 expressions.

Even though, the result (accuracy score of 0,58289 in Kaggle competition) was lower than the three approaches examined in (Athanasia Koumpouri, Iosif Mporas and Vasileios Megalooikonomou, “Evaluation of Four Approaches for Sentiment Analysis on Movie Reviews”, in the proceedings of Kaggle Competition), it was slightly higher than the fourth approach that is Statistical-based Approach (STA) (accuracy score of 0.56802).

The datasets have been collected and aggregated from the website of Kaggle. (Available at: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)
• train.tsv consists of the expressions and their related sentiment labels. Moreover, it has been given a SentenceId with the goal that we can track which phrases have a place with a solitary sentence.
• test.tsv contains only expressions and we should allocate a sentiment label to every single expression.
In addition, the sentiment labels are:
0 - negative,
1 - somewhat negative,
2 - neutral,
3 - somewhat positive,
4 - positive.
