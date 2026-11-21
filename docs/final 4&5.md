**EXPERIMENTAL TESTING GUIDE**

*Financial-Related Smishing Detection Using a Hybrid CNN–Random Forest Model with FiLM*

Purpose: This guide will serve as the group's reference when conducting the experiments, evaluating the models, and preparing the results for Chapters 4 and 5\.

# **I. OVERVIEW OF THE TESTING PROCESS**

Follow these steps in order:

| Step | Activity | Expected Output |
| :---- | :---- | :---- |
| 1 | Finalize the dataset | Final verified dataset |
| 2 | Preprocess SMS and extract URL features | Model-ready dataset |
| 3 | Prepare cross-validation splits | Five-fold assignments |
| 4 | Train the four main models | Trained models and logs |
| 5 | Generate predictions | Out-of-fold predictions |
| 6 | Conduct component checks | Component-check results |
| 7 | Calculate performance metrics | Confusion matrices and classification metrics |
| 8 | Analyze message conditions | Performance per message condition |
| 9 | Conduct error analysis | Classification error records |
| 10 | Perform McNemar's test | Statistical comparison results |

 

# **II. STEP-BY-STEP EXPERIMENTAL PROCEDURES**

## **STEP 1\. FINALIZE THE DATASET**

What to do:

1\.      Consolidate the eligible SMS records from the authorized datasets.

2\.      Remove exact duplicates and review near-duplicates.

3\.      Apply the inclusion and exclusion criteria.

4\.      Complete the researchers' independent annotations.

5\.      Complete external validation using the agreed sample.

6\.      Resolve disagreements through discussion and document the final labels.

7\.      Finalize the Smishing and Legitimate classifications.

8\.      Freeze the dataset before model training.

Results to obtain:

| Dataset Information | Result |
| :---- | :---- |
| Total retained messages | 6718 |
| Smishing messages | 3097 |
| Legitimate messages | 3621 |
| Messages with URLs | 5,830 (3,022 smishing, 2,808 legitimate) |
| Messages without URLs | 888 (75 smishing, 813 legitimate) |
| Accessible webpages | 3,624 (1,719 smishing, 1,905 legitimate) |
| Inaccessible webpages | 2,206 (1,303 smishing, 903 legitimate) |

 

Also record the distribution of messages across the six financial-related categories.

Files to save:

·        Final Dataset

·        Dataset Statistics

·        Annotation and Validation Results

Before proceeding: Verify that all messages have unique IDs and finalized labels.

## **STEP 2\. PREPROCESS SMS AND EXTRACT URL FEATURES**

What to do:

**A. SMS Text Preprocessing**

9\.      Extract the SMS text.

10\.   Separate and remove embedded URLs from the text.

11\.   Convert the text to lowercase.

12\.   Normalize repeated punctuation.

13\.   Tokenize the messages.

14\.   Pad or truncate messages to 50 tokens.

15\.   Convert tokens using pretrained English and Tagalog fastText embeddings.

16\.   Apply the established language and subword fallback procedure.

The vocabulary and other learned preprocessing components must be fitted using training data only.

**B. URL Feature Extraction**

For messages containing URLs, extract the 37 URL structural features specified in Appendix 4\. The features cover URL-string characteristics, webpage content, and domain registration and ranking information.

| Message Condition | Treatment |
| :---- | :---- |
| No URL | has\_url \= 0; all 37 URL features \= \-1 |
| URL exists but webpage is inaccessible | Retain obtainable URL and domain features; webpage features \= \-1; page\_missing \= 1 |
| URL and webpage accessible | Extract the available features; page\_missing \= 0 |

 

For messages without URLs, FiLM modulation is bypassed using γ \= 1 and β \= 0\.

Results to obtain:

·        Number of messages with and without URLs.

·        Number of accessible and inaccessible webpages.

·        Completed text preprocessing.

·        Completed 37 URL features.

·        Correct missing-value assignments.

Files to save:

·        Preprocessed SMS Text

·        URL Structural Features

·        Feature Extraction Log

·        Final Model Input

Before proceeding: Check that the text and URL features correspond to the correct message IDs.

## **STEP 3\. PREPARE THE CROSS-VALIDATION SPLITS**

What to do:

Your study uses Stratified Group 5-Fold Cross-Validation with three random seeds: 42, 123, and 2024\.

17\.   Divide the dataset into five folds for each seed.

18\.   Maintain class representation through stratification.

19\.   Keep related messages together through grouping.

20\.   Use four folds for training and model development.

21\.   Reserve the remaining fold as the held-out test set.

22\.   Repeat until all five folds have served as the test set.

| Random Seed | Number of Folds |
| :---- | :---- |
| 42 | 5 |
| 123 | 5 |
| 2024 | 5 |
| Total | 15 |

 

Each model undergoes 15 held-out fold evaluations.

Important: The test fold must not be used for training, early stopping, or model selection. Any validation data must come from the training portion.

File to save: Cross\_Validation\_Splits.xlsx

Before proceeding: Verify that every message appears in exactly one test fold per seed and that related messages remain in the same fold.

## **STEP 4\. TRAIN THE FOUR MAIN MODELS**

What to do:

Train and evaluate the following four configurations:

| Model | Main Configuration | Purpose |
| :---- | :---- | :---- |
| CNN-only | SMS text \+ 37 URL features | Baseline without Random Forest |
| Concatenation Fusion | CNN features \+ Random Forest outputs | Conventional feature fusion baseline |
| KNN-RF-ETC | Text ensemble \+ separate URL classifier | External comparative baseline |
| Proposed Model | CNN \+ Random Forest \+ FiLM | Proposed architecture |

 

For the proposed model:

23\.   Train the Random Forest using the 37 URL features from the training portion.

24\.   Generate the smishing probability (P) and uncertainty (u).

25\.   Obtain the separate page\_missing indicator.

26\.   Train the CNN, trainable fastText embeddings, and FiLM Generator jointly.

27\.   Use the specified class-weighted loss, early stopping, and modality dropout.

The Random Forest is fitted separately and is not jointly optimized with the CNN and FiLM Generator.

Repeat the training procedure for all five folds and three seeds.

Results to record:

·        Training and validation performance.

·        Number of training and validation messages.

·        Best epoch and early stopping information.

·        Model configuration, random seed, and fold.

·        Training completion or failure.

·        Saved model checkpoints.

Files to save: Training Logs, Model Checkpoints, and Model Configuration Records.

Before proceeding: Confirm that all four models have completed their required runs.

## **STEP 5\. GENERATE OUT-OF-FOLD PREDICTIONS**

What to do:

After training each model, use it to classify the corresponding held-out test messages. Save the prediction of every individual message.

Required columns:

| Column | Description |
| :---- | :---- |
| Message ID | Unique message identifier |
| Actual Label | Ground-truth classification |
| Predicted Label | Model prediction |
| Predicted Probability | Probability, where applicable |
| Model | Model configuration |
| Seed | Random seed |
| Fold | Test fold |

 

Each model must have three complete prediction sets: seed 42, seed 123, and seed 2024\. Each set must contain every message in the evaluation dataset exactly once.

Files to save:

·        CNN-only Predictions

·        Concatenation Predictions

·        KNN-RF-ETC Predictions

·        Proposed Model Predictions

Important: Do not save only accuracy or F1-score. Individual predictions are required for confusion matrices and McNemar's test.

## **STEP 6\. CONDUCT THE FOUR COMPONENT CHECKS**

What to do:

Evaluate the proposed model under four additional configurations to examine the contribution of its components.

| Component Check | What Changes | Purpose |
| :---- | :---- | :---- |
| Text-only CNN | Uses text without URL conditioning | Examine text-based performance |
| FiLM Neutralized | γ \= 1 and β \= 0 | Examine the effect of disabling FiLM |
| Shuffled RF Outputs | RF outputs are shuffled across messages | Examine correct URL-message correspondence |
| Webpage Features Hidden | Webpage information is hidden | Examine the contribution of webpage features |

 

Results to obtain for each configuration:

·        Accuracy

·        Precision

·        Recall

·        F1-score

Use the corresponding cross-validation framework and document the exact configuration of each check.

File to save: Component\_Check\_Results.xlsx

Keep the results for each seed before calculating the overall mean and standard deviation.

## **STEP 7\. CALCULATE THE CLASSIFICATION PERFORMANCE METRICS**

What to do:

Using the saved out-of-fold predictions, generate a confusion matrix for each of the four main models. The positive class is Smishing.

| Actual / Predicted | Smishing | Legitimate |
| :---- | :---- | :---- |
| Smishing | TP | FN |
| Legitimate | FP | TN |

 

Calculate the following four metrics:

| Metric | Formula |
| :---- | :---- |
| Accuracy | (TP \+ TN) / (TP \+ TN \+ FP \+ FN) |
| Precision | TP / (TP \+ FP) |
| Recall | TP / (TP \+ FN) |
| F1-score | 2 × (Precision × Recall) / (Precision \+ Recall) |

 

Calculate each metric separately for seeds 42, 123, and 2024\. Then calculate the mean and standard deviation across the three seeds.

| Model | Accuracy | Precision | Recall | F1-score |
| :---- | :---- | :---- | :---- | :---- |
| CNN-only | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] |
| Concatenation Fusion | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] |
| KNN-RF-ETC | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] |
| Proposed Model | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] | \[Mean ± SD\] |

 

File to save: Classification\_Performance\_Results.xlsx. Include the individual seed results and confusion matrices.

## **STEP 8\. ANALYZE PERFORMANCE BY MESSAGE CONDITION**

What to do:

Examine the performance of the models under three message conditions:

28\.   Messages without URLs.

29\.   Messages with inaccessible webpages.

30\.   Messages with accessible webpages.

Calculate the number of messages, accuracy, precision, recall, and F1-score for each condition.

| Message Condition | Number of Messages | Accuracy | Precision | Recall | F1-score |
| :---- | :---- | :---- | :---- | :---- | :---- |
| No URL | \[N\] | \[ \] | \[ \] | \[ \] | \[ \] |
| Inaccessible webpage | \[N\] | \[ \] | \[ \] | \[ \] | \[ \] |
| Accessible webpage | \[N\] | \[ \] | \[ \] | \[ \] | \[ \] |

 

Keep the results per seed and identify the aggregation method.

File to save: Message\_Condition\_Analysis.xlsx

## **STEP 9\. CONDUCT CLASSIFICATION ERROR ANALYSIS**

What to do:

Examine correctly and incorrectly classified messages to identify possible patterns in the model's predictions.

| Classification | Meaning | What to Examine |
| :---- | :---- | :---- |
| TP | Smishing correctly detected | Suspicious wording and URL characteristics |
| TN | Legitimate correctly detected | Normal message patterns and legitimate links |
| FP | Legitimate classified as smishing | Urgent wording, financial terms, OTP phrases |
| FN | Smishing classified as legitimate | Neutral wording, missing URL evidence, inaccessible webpages |

 

Select representative anonymized messages and document their characteristics. You do not need to examine every message individually in the manuscript. Focus on recurring patterns.

File to save: Classification\_Error\_Analysis.xlsx

## **STEP 10\. PERFORM McNEMAR'S TEST**

What to do:

Compare the proposed model with each of the three baseline models:

31\.   Proposed Model vs. CNN-only

32\.   Proposed Model vs. Concatenation Fusion

33\.   Proposed Model vs. KNN-RF-ETC

Use paired out-of-fold predictions from the same messages and the same random seed.

Identify two values:

·        b: Messages correctly classified by the proposed model but incorrectly classified by the baseline.

·        c: Messages incorrectly classified by the proposed model but correctly classified by the baseline.

Calculate the continuity-corrected McNemar statistic:

**χ² \= (\|b − c\| − 1)² / (b \+ c)**

Record b, c, the chi-square statistic, p-value, and statistical decision.

Significance level: α \= 0.05

| p-value | Decision |
| :---- | :---- |
| Below 0.05 | Reject H₀ |
| Equal to or above 0.05 | Fail to reject H₀ |

 

A significant result indicates a difference in classification outcomes. Examine b and c to determine which model has more correct classifications among the disagreement cases.

Repeat all three comparisons for all three seeds, for a total of nine statistical tests.

Confirm with your adviser or statistician whether a multiple-comparison adjustment is needed.

File to save: McNemar\_Test\_Results.xlsx

# **III. FINAL CHECKLIST BEFORE WRITING CHAPTERS 4 AND 5**

## **Dataset and Preprocessing**

☐  Final dataset is verified.

☐  Annotations and external validation are completed.

☐  Dataset statistics are available.

☐  Text preprocessing is completed.

☐  All 37 URL features are extracted or properly assigned missing values.

## **Cross-Validation and Training**

☐  Five folds are completed for seed 42\.

☐  Five folds are completed for seed 123\.

☐  Five folds are completed for seed 2024\.

☐  All four main models are trained and evaluated.

☐  Training logs and model checkpoints are saved.

☐  Individual out-of-fold predictions are saved.

## **Additional Experiments**

☐  Four component checks are completed.

☐  Message-condition analysis is completed.

☐  Classification error analysis is completed.

## **Performance and Statistical Results**

☐  Confusion matrices are available.

☐  Accuracy is calculated.

☐  Precision is calculated.

☐  Recall is calculated.

☐  F1-score is calculated.

☐  Mean and standard deviation are calculated.

☐  All nine McNemar's tests are completed.

☐  Statistical decisions are verified.

# **IV. HOW THE RESULTS WILL BE USED IN CHAPTERS 4 AND 5**

## **CHAPTER 4 – RESULTS AND DISCUSSION**

| Suggested Section | Results to Present |
| :---- | :---- |
| 4.1 Dataset Composition | Final dataset counts, class distribution, financial categories, and URL availability |
| 4.2 Model Performance | Confusion matrices, accuracy, precision, recall, and F1-score |
| 4.3 Component Checks | Results of the four additional configurations |
| 4.4 Message-Condition Analysis | Performance on messages with no URL, inaccessible webpages, and accessible webpages |
| 4.5 Classification Error Analysis | TP, TN, FP, and FN patterns |
| 4.6 McNemar's Test | Nine comparisons, p-values, and statistical decisions |

 

## **CHAPTER 5 – SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS**

5.1 Summary of Findings: Summarize the important results obtained from the experiments.

5.2 Conclusions: Answer the research questions based on the actual findings. Discuss the performance of the proposed model, its comparison with the baselines, and the statistical results.

5.3 Recommendations: Provide recommendations based on the limitations and findings of the study, particularly concerning dataset diversity, URL feature extraction, model improvements, and future evaluation.

# **V. IMPORTANT ITEMS TO CLARIFY BEFORE FINAL TESTING**

34\.   Dataset totals: Chapter 3, Table 1, contains an inconsistency in the legitimate-message total. Based on the displayed source counts, 1,115 \+ 2,500 \= 3,615, not 3,665.

35\.   Dataset sources: Clarify how the eight publicly available datasets described in the data-gathering procedure relate to the ISDA and crowdsourced datasets described in the Sources of Data section.

36\.   ISDA's role: Confirm whether ISDA is used only for text-component development or is also included in the final evaluation dataset.

37\.   KNN-RF-ETC: Confirm the training sources and how the text and URL classifier outputs are combined.

38\.   Grouping: Confirm which identifiers are used to keep related messages and campaign templates in the same fold.

39\.   McNemar's test: Confirm the multiple-comparison procedure with your adviser or statistician.

These must be settled before final testing to avoid inconsistencies between the actual results and Chapter 3\.

# **FINAL REMINDER**

**Follow the ten experimental steps in order. Save the actual predictions, training logs, and statistical outputs instead of keeping only screenshots or summarized results.**

