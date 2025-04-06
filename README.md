# News Article Classification: Naive Bayes vs LLM Approaches

This project compares traditional machine learning (Naive Bayes) and modern LLM-based approaches for classifying news articles into categories. It uses the AG News dataset to demonstrate different classification techniques and evaluates their performance.

## Project Overview

The project consists of two main parts:
1. **Traditional ML Classification**: Using Naive Bayes to classify news articles
2. **LLM-based Classification**: Using Llama 3.2 (1B parameter model) with both zero-shot and few-shot approaches

## Dataset

The AG News dataset contains news articles classified into 4 categories:
- World
- Sports
- Business
- Science/Technology

Dataset statistics:
- Training samples: 120,000
- Testing samples: 7,600

## Methodologies

### Naive Bayes Classification
- Used `CountVectorizer` with English stop words removed for feature extraction
- Implemented `MultinomialNB` from scikit-learn
- Evaluated with accuracy metrics and classification report

### LLM-based Classification
1. **Zero-Shot Learning**:
   - The model was provided with article text and asked to classify without examples
   - Used a simple prompt structure asking for category prediction

2. **Few-Shot Learning**:
   - The model was provided with 3 example article-category pairs
   - Examples demonstrated how to classify articles across different categories

## Results

### Naive Bayes Classifier
- **Accuracy**: 90.4%
- **Performance by category**:
  - Category 1 (World): Precision: 0.91, Recall: 0.90, F1-score: 0.91
  - Category 2 (Sports): Precision: 0.95, Recall: 0.98, F1-score: 0.97
  - Category 3 (Business): Precision: 0.87, Recall: 0.85, F1-score: 0.86
  - Category 4 (Science/Tech): Precision: 0.88, Recall: 0.89, F1-score: 0.88

### LLM Results (Llama 3.2-1B)

#### Zero-Shot Learning
Performance was suboptimal:
- Most articles were classified as "World" regardless of content
- Example: "Oil prices rise due to increased demand" → World (should be Business)
- Example: "Local team wins championship" → World (should be Sports)

#### Few-Shot Learning
Performance improved with examples:
- Example: "Oil prices rise due to increased demand" → Business (correct)
- Example: "Local team wins championship" → World (incorrect)
- Example: "New study reveals health benefits of exercise" → Business (incorrect)
- Example: "Global leaders meet for climate summit" → World (correct)

## Conclusion

The Naive Bayes classifier demonstrated strong performance with 90.4% accuracy, outperforming the LLM-based approaches in this experiment. The few-shot learning approach with the LLM showed some improvement over zero-shot learning but still failed to match the traditional ML approach.

This project demonstrates that while LLMs have significant capabilities, traditional ML approaches like Naive Bayes remain highly effective for specialized classification tasks when properly trained on domain-specific data.

## Technologies Used

- Python
- scikit-learn (for Naive Bayes implementation)
- HuggingFace Transformers (for LLM implementation)
- Pandas (for data handling)
- Llama 3.2-1B (for LLM classification)

## Future Work

- Evaluate performance with larger LLM models (e.g., 7B, 13B parameters)
- Test more sophisticated prompt engineering techniques
- Experiment with fine-tuning LLMs on the AG News dataset
- Compare with other traditional ML approaches (SVM, Random Forest, etc.)
- Implement a hybrid approach combining traditional ML and LLMs