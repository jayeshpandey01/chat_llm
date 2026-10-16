HealthTrack AI - Patient Symptom Checker

Build a web application that allows users to input their symptoms and receive potential health conditions based on machine learning models.

About 6 hours of work, due within 4 days.
Python · AIML · LLM · Gen Ai

HealthTrack AI is a fictional telehealth company that aims to empower patients by providing an AI-driven symptom checker. The primary user is a patient seeking preliminary insights into their health concerns. A good submission demonstrates clean code structure, effective error handling, meaningful commit messages, and a clear README that documents the architecture and trade-offs made during development.


Implement a symptom input form where users can enter their symptoms as free text.
How it’s checked
Given the user accesses the form, when they submit symptoms, then the input should be captured and processed.
Must
Develop a machine learning model using a pre-existing dataset to predict potential health conditions based on the symptoms provided.
How it’s checked
Given the model is trained, when symptoms are inputted, then the model should return a list of potential health conditions with confidence scores.
Must
Create a results page that displays the predicted health conditions along with explanations for each condition based on the symptoms.
How it’s checked
Given the model outputs conditions, when the results page is loaded, then the predicted conditions and explanations should be displayed clearly.
Should
Implement data preprocessing steps to clean and prepare the symptom data before feeding it into the model.
How it’s checked
Given the raw symptom data, when preprocessing is applied, then the data should be cleaned and formatted appropriately for model input.
Should
Add a caching layer to store frequently queried symptoms and their results to improve response times.
How it’s checked
Given repeated queries for the same symptoms, when the symptoms are queried, then the response time should be significantly reduced.
Must
Document the architecture and any trade-offs made in the design of the application in the README file.
How it’s checked
Given the README file, when reviewed, then it should clearly outline the architecture, trade-offs, and rationale behind design decisions.
Implement user authentication to allow users to save their symptom history and predictions.
How it’s checked
Given a user account, when they log in, then they should be able to view their saved symptom history.