# TrueInsight — Fake Review Detection and Product Genuinity Analyzer

> A web-based e-commerce application that detects potentially fake and suspicious product reviews using machine learning, sentiment analysis, behavioral analysis, and review pattern detection.

---

## Overview

**TrueInsight** is a web-based Fake Review Detection and Product Genuinity Analyzer developed to improve the reliability and transparency of product reviews in e-commerce platforms.

Online reviews have a major influence on purchasing decisions, but fake, duplicated, manipulated, or misleading reviews can significantly affect product ratings and make it difficult for users to determine the actual quality of a product.

TrueInsight addresses this problem by analyzing product reviews using a combination of:

- Machine Learning
- Sentiment Analysis
- Behavioral Analysis
- Duplicate Review Detection
- Rating Pattern Analysis
- User Activity Analysis

The system uses a **Random Forest Classifier** as the primary machine learning model. **TextBlob** is used for sentiment analysis, while additional features such as review length, rating deviation, duplicate content, and user activity are used to identify suspicious review patterns.

The system also provides a **dual-rating mechanism**, displaying both the original product rating and a filtered rating calculated after excluding reviews classified as suspicious.

---

## Problem Statement

Traditional e-commerce review systems generally treat all submitted reviews equally when calculating product ratings.

This creates several problems:

- Fake positive or negative reviews can manipulate product ratings.
- Duplicate reviews can artificially increase review volume.
- Abnormal user activity can indicate review manipulation.
- Sentiment and rating inconsistencies can indicate suspicious reviews.
- Users have no clear way to determine whether a review is genuine.
- The final product rating may not accurately represent genuine customer opinions.

Even a relatively small number of fake reviews can influence the overall rating of a product.

TrueInsight aims to reduce this problem by automatically analyzing reviews and identifying potentially suspicious reviews before they significantly affect the product's filtered rating.

---

## Proposed Solution

TrueInsight combines textual, rating-based, and behavioral information to classify reviews.

The overall approach is:

```text
                  Product Review
                        |
                        v
                Feature Extraction
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
   Text Features   Sentiment       User Behaviour
        |           Analysis            |
        |               |               |
        +---------------+---------------+
                        |
                        v
              Duplicate Detection
                        |
                        v
               Random Forest Model
                        |
                        v
              Prediction Probability
                        |
                        v
              Threshold Classification
                        |
              +---------+---------+
              |                   |
              v                   v
          Genuine             Suspicious
              |                   |
              +---------+---------+
                        |
                        v
                 Rating Calculation
                        |
              +---------+---------+
              |                   |
              v                   v
          Raw Rating       Filtered Rating
