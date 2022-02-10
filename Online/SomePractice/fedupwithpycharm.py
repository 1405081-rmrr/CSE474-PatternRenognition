import numpy
import pandas
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
from bs4 import BeautifulSoup
with open("Test.txt") as f:
    for line in f:
        print(line)