# TrueInsight

### Fake Review Detection and Product Genuinity Analyzer

TrueInsight is a web-based e-commerce application that analyzes product reviews and identifies potentially fake or suspicious reviews using machine learning, sentiment analysis, and behavioral patterns.

The system is designed to reduce the effect of misleading reviews on product ratings by analyzing submitted reviews in real time and providing both **raw ratings** and **filtered ratings**.

---

## Overview

Online product reviews have a major influence on purchasing decisions. However, fake, manipulated, duplicate, or misleading reviews can distort product ratings and make it difficult for users to judge the actual quality of a product.

TrueInsight addresses this problem by combining:

- Machine Learning
- Sentiment Analysis
- Behavioral Analysis
- Duplicate Review Detection
- Rating Analysis
- Real-time Review Classification

A **Random Forest classifier** is used as the primary machine learning model. Review text is additionally analyzed using **TextBlob** to identify sentiment inconsistencies between the review content and the assigned rating.

The system then classifies reviews as either **Genuine** or **Suspicious** and calculates a filtered product rating by excluding suspicious reviews.

---

## Key Features

- User registration and login
- Product browsing
- Product search
- Product reviews and ratings
- Fake review detection
- Random Forest based classification
- Review sentiment analysis using TextBlob
- Duplicate review detection
- User behavior analysis
- Rating inconsistency detection
- Real-time review classification
- Suspicious review identification
- Raw rating calculation
- Filtered rating calculation
- Review classification output
- MySQL database integration
- Web-based user interface

---

## How It Works

The system follows a simple processing pipeline:

```text
User
  |
  v
Web Interface
  |
  v
Flask Backend
  |
  v
MySQL Database
  |
  v
Review Retrieval
  |
  v
Feature Engineering
  |
  +----------------------+
  |                      |
  v                      v
Sentiment Analysis    Behavioral Analysis
  |                      |
  +----------+-----------+
             |
             v
      Random Forest Model
             |
             v
      Prediction Probability
             |
             v
      Classification Logic
             |
       +-----+-----+
       |           |
       v           v
   Genuine     Suspicious
       |           |
       +-----+-----+
             |
             v
     Rating Calculation
             |
       +-----+-----+
       |           |
       v           v
   Raw Rating  Filtered Rating
             |
             v
        Web Interface
