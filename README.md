# Hate Speech Detection (v 1.0)
![GitHub followers](https://img.shields.io/github/followers/PearlyShah?style=social)
![Profile views](https://gpvc.arturio.dev/PearlyShah)

## Overview
The main goal of this project is to identify hate speech.For that models like k-nearest neighbour,logistic regression,random forest classifier,support vector machine and multinomial naive bayes classifier models are used in this project.The feature used is count vectorizer.Here every models have almost same accuracy but the random forest classifier have the highest accuracy of 93%.

## Execution of the project

- Download data.csv
- Download and run Hate Speech Detection.ipynb

## Methodology

The project is implemented in four steps:
1. Loading the data
2. Pre-Processing the data 
3. Feature Implementation
4. Build and Train Model

![Untitled Diagram (1)](https://user-images.githubusercontent.com/43084772/129882780-e2f25376-0033-4af3-b1a7-15ce130db1cf.jpg)

## Dependencies

1. pandas 
2. numpy 
3. nltk
4. string
5. word_tokenize
6. stopwords
7. WordNetLemmatizer



## Analysis:
The data is collected from 'Toxic Comment Classification Challenge' from the kaggle competation.Where each comments are labeled with the class of toxicity i.e toxic,severe toxic,obscene,threat,insult,identity hate.The data is multilabeled i.e each comment contains more than one class.
Below is the count of each class for the number of comments.

![count_class](https://user-images.githubusercontent.com/43084772/129882999-3e1e197c-d8fe-4cc6-b04a-b63af1698236.PNG)

## Results
After preprocessing,feature implementation ,building and training model.The result obtained is as follow:

![re_result](https://user-images.githubusercontent.com/43084772/129883077-36611d2a-d639-42ea-8fc0-6bdb36a44f0b.PNG)

### Comparition between models
![knn](https://user-images.githubusercontent.com/43084772/129886157-9061e7d8-9e10-4580-a168-201b4c4a8618.PNG)
![logistic](https://user-images.githubusercontent.com/43084772/129886243-b0d7ff77-779a-4b01-b944-33c24f39b992.PNG)
![nb](https://user-images.githubusercontent.com/43084772/129886245-a6b8a490-1ae9-4272-88c6-d9a17e915cf5.PNG)
![rfc](https://user-images.githubusercontent.com/43084772/129886246-ecbb64ff-66fa-406a-9b9b-95f57473d6ba.PNG)
![svm](https://user-images.githubusercontent.com/43084772/129886249-d357b05e-406c-4966-b645-31434dc14349.PNG)

## Version History
- 1.0
    - Final Release

## License
The Hate speech detection project is not licensed

## Contact
- Pearly Shah
- Email id: pearlyshahs@gmail.com




