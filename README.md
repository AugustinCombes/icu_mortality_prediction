# icu_mortality_prediction
Mortality prediction using MIMIC-III dataset :

This project makes use of MIMIC-III patient records to predict whether they will die during their ICU stay.
The data used is the three first hours of event of ICU stays which duration is between 24 and 48 hours.

---

### Running the code
1. You will first need to request access for MIMIC-III, a publicly avaiable electronic health records collected from ICU patients over 11 years.
2. Then, create the preprocessed *events.json* and *targets.json* files using the *preprocessMIMIC.py* script: `python3 preprocessMIMIC.py <raw_mimic_path>` where the folder of path *<raw_mimic_path>* contains the files *ICUSTAYS.csv*, *ADMISSIONS.csv* and *CHARTEVENTS.csv*.
3. Finally, you can run the inference using the *main.ipynb* notebook.

---

### Methodology

#### Data processing:
- The preprocess script uses [vaex](https://vaex.io/)'s lazy computation to efficiently perform filtering on the raw data.
Then, it uses dictionary logic to group the EHR codes which total occurences are less than 100 in an "other" token, and saves the result as json files, both for the events and the labels.
- The train-test split is done in the *dataloader.py* script, where ICU admission records whose ICUSTAY_ID ends with the digit 8 & 9 are left out as a test set, while the rest of the samples as the training set.

#### Model:
- The inference model is a two-layers transformers. It uses a custom minutes-based positional encoding to address that the ICU events typically occur in packed series (for instance 5 events at minute 18, then 8 events at minute 93, etc). Therefore, the sinusoidal encoding is indexed by the minute-timesteps of the events instead by the sequence index.
- This positional encoding is concatenated, along with the ICU events codes embedding and the ICU events values, to produce an embedded icu stay representation, ready for downstream inference.
- The imbalance of the data (93% survival rate) is addressed with oversampling of the minority class during the training inference.

#### Performance:
- In the context of imbalanced binary classification, the loss used is the Binary Cross-Entropy, and the main metric of interest is AUPRC, which summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
- Training the model on 50 epochs, using a 48-dimensional patient representation and a 128-dimensional two-layers transformers typically lead to AUPRC test scores between *0.35* and *0.40*, the best I have observed so far being *0.43*.
