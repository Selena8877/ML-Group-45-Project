| Language | Model | Configs tried | Selected configuration | Val Macro-F1 | Test Macro-F1 [95% CI] |
|---|---|---|---|---|---|
| Arabic | KNN (CountVectorizer) | 6 | counts, k=9, uniform | 0.831 | 0.838 [0.761–0.900] |
| Arabic | Multinomial Naive Bayes | 6 | counts, alpha=2 | 0.941 | 0.900 [0.839–0.950] |
| Arabic | KNN (tuned) | 28 | tfidf, k=25, distance | 0.914 | 0.850 [0.779–0.910] |
| Arabic | Logistic Regression | 28 | tfidf, C=10, class_weight=none | 0.943 | 0.950 [0.910–0.990] |
| Chinese | KNN (CountVectorizer) | 6 | counts, k=11, uniform | 0.666 | 0.788 [0.706–0.860] |
| Chinese | Multinomial Naive Bayes | 6 | counts, alpha=5 | 0.699 | 0.859 [0.795–0.920] |
| Chinese | KNN (tuned) | 28 | tfidf, k=41, uniform | 0.711 | 0.838 [0.767–0.909] |
| Chinese | Logistic Regression | 28 | tfidf, C=1, class_weight=none | 0.714 | 0.849 [0.777–0.910] |
| Portuguese | KNN (CountVectorizer) | 6 | counts, k=3, uniform | 0.616 | 0.558 [0.462–0.653] |
| Portuguese | Multinomial Naive Bayes | 6 | counts, alpha=0.5 | 0.798 | 0.604 [0.510–0.696] |
| Portuguese | KNN (tuned) | 28 | tfidf, k=5, uniform | 0.668 | 0.517 [0.422–0.609] |
| Portuguese | Logistic Regression | 28 | tfidf, C=10, class_weight=balanced | 0.763 | 0.655 [0.563–0.743] |
| Thai | KNN (CountVectorizer) | 6 | counts, k=9, uniform | 0.628 | 0.454 [0.376–0.540] |
| Thai | Multinomial Naive Bayes | 6 | counts, alpha=0.1 | 0.762 | 0.618 [0.524–0.700] |
| Thai | KNN (tuned) | 28 | tfidf, k=9, uniform | 0.663 | 0.454 [0.376–0.524] |
| Thai | Logistic Regression | 28 | tfidf, C=10, class_weight=balanced | 0.784 | 0.736 [0.644–0.815] |

Search spaces (all use character 2–3 grams fitted on train only; selection by validation Macro-F1, seed 42):

- **KNN (CountVectorizer)**: counts; k ∈ {1,3,5,7,9,11}; uniform; cosine (6 configurations)
- **Multinomial Naive Bayes**: counts; alpha ∈ {0.01,0.1,0.5,1,2,5} (6 configurations)
- **KNN (tuned)**: counts|tfidf × weights ∈ {uniform,distance} × k ∈ {1,3,5,9,15,25,41}; cosine (28 configurations)
- **Logistic Regression**: counts|tfidf × class_weight ∈ {none,balanced} × C ∈ {0.001,…,1000} (28 configurations)
