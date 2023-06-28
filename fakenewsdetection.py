import pandas as pd     
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Read the fake and true datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Add a "class" column to indicate fake (0) or true (1)
data_fake["class"] = 0
data_true["class"] = 1

# Remove unnecessary rows from the datasets
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)

# Concatenate the fake and true datasets
data_merge = pd.concat([data_fake, data_true], axis=0)

# Remove unwanted columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Check for null values
data.isnull().sum()

# Shuffle the data
data = data.sample(frac=1)

# Reset the index
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# Define a function for text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text preprocessing to the text column
data['text'] = data['text'].apply(wordopt)

# Split the data into input features (x) and target variable (y)
x = data['text']
y = data['class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Create a TfidfVectorizer object for text vectorization
vectorization = TfidfVectorizer()

# Fit and transform the training data
xv_train = vectorization.fit_transform(x_train)

# Transform the testing data
xv_test = vectorization.transform(x_test)

print(xv_train)
print(xv_test)


# Save the training and testing data using pickle
pickle.dump(x_train, open('x_train.sav', 'wb'))
pickle.dump(x_test, open('x_test.sav', 'wb'))

# Logistic Regression model
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Save the Logistic Regression model
pickle.dump(LR, open('lrModel.sav', 'wb'))

# Make predictions using the Logistic Regression model
pred_lr = LR.predict(xv_test)

# Print the classification report for the Logistic Regression model
print(classification_report(y_test, pred_lr))

# Decision Tree Classifier model
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

# Save the Decision Tree Classifier model
pickle.dump(DT, open('dtModel.sav', 'wb'))

# Make predictions using the Decision Tree Classifier model
pred_dt = DT.predict(xv_test)

# Print the accuracy score for the Decision Tree Classifier model
print(DT.score(xv_test, y_test))

# Print the classification report for the Decision Tree Classifier model
print(classification_report(y_test, pred_dt))

# Gradient Boosting Classifier model
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

# Save the Gradient Boosting Classifier model
pickle.dump(GB, open('gbModel.sav', 'wb'))

# Make predictions using the Gradient Boosting Classifier model
predict_gb = GB.predict(xv_test)

# Print the accuracy score for the Gradient Boosting Classifier model
print(GB.score(xv_test, y_test))

# Print the classification report for the Gradient Boosting Classifier model
print(classification_report(y_test, predict_gb))

# Random Forest Classifier model
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

# Save the Random Forest Classifier model
pickle.dump(RF, open('rfModel.sav', 'wb'))

# Make predictions using the Random Forest Classifier model
predict_rf = RF.predict(xv_test)

# Print the accuracy score for the Random Forest Classifier model
print(RF.score(xv_test, y_test))

# Print the classification report for the Random Forest Classifier model
print(classification_report(y_test, predict_rf))

# Define a function to output the label based on the predicted class
def output_label(n):
    if n == 0:
        return 'Fake News'
    elif n == 1:
        return 'Not a fake news'

# Define a function for manual testing
def manual_testing(news):
    testing_news = {'text': [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GB.predict(new_xv_test)
    pred_RFC = RF.predict(new_xv_test)

    print('\n\nPrediction: {}\nLR Prediction: {}\nDT Prediction: {}\nGB Prediction: {}\nRF Prediction: {}'.format(news,
                                                                                                                    output_label(pred_LR[0]),
                                                                                                                    output_label(pred_DT[0]),
                                                                                                                    output_label(pred_GBC[0]),
                                                                                                                    output_label(pred_RFC[0])))

# Interactive loop for manual testing
while True:
    news = str(input('Enter the news: '))
    manual_testing(news)

