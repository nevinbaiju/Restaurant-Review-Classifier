from django.http import HttpResponse
from django.shortcuts import render
from . import classifier
# Create your views here.
def index(request):
	return render(request, 'review_classifier/index.html', {})
def classify(request):
	review = request.POST.get("review", "")
	p = classifier.predictor()
	result = p.predict(review)
	return render(request, 'review_classifier/result.html', {'avg_result' : result})