
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class predictor():

	def __init__(self):
		self.load_pickles()

	def load_pickles(self):
		with open('/home/nevin/Nevin/projects/first-django/mysite/review_classifier/pickles/Decision_Tree.pkl', 'rb') as file:
			self.DTree = pickle.load(file)
		with open('/home/nevin/Nevin/projects/first-django/mysite/review_classifier/pickles/Gaussian_model.pkl', 'rb') as file:
			self.gaussianNB = pickle.load(file)
		with open('/home/nevin/Nevin/projects/first-django/mysite/review_classifier/pickles/Random_Forest.pkl', 'rb') as file:
			self.RForest = pickle.load(file)
		with open('/home/nevin/Nevin/projects/first-django/mysite/review_classifier/pickles/count_vectoriser.pkl', 'rb') as file:
			self.cv = pickle.load(file)

	def clean_string(self, review):
		review = re.sub('[^a-zA-Z]', ' ', review)
		review = review.lower()
		review = review.split()
		ps = PorterStemmer()
		review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		return review

	def predict(self, review):
		review = self.clean_string(review)
		vector = self.cv.transform([review]).toarray()

		result = []
		result.append(self.gaussianNB.predict(vector)[0])
		result.append(self.DTree.predict(vector)[0])
		result.append(self.RForest.predict(vector)[0])
		avg_result = np.array(result).mean()

		print(avg_result)
		print(result)

		if(avg_result>0.5):
			print("The review is good")
		else:
			print("The review is bad")
		return avg_result
