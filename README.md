# Spotify Playlist Curator - Machine Learning Mini-Project

## 1. Project overview

This project is a **supervised multiclass classification prototype** that predicts which of 13 selected playlist classes a song belongs to from its audio characteristics and related metadata. It uses the supplied TidyTuesday `spotify_songs.csv` dataset and compares paper-inspired baseline models with an extended feature set and tree-based ensembles.

The work is inspired by Amel Awadelkarim and Kevin Coelho (2018), *Training a Playlist Curator Based on User Taste*. It is a **paper-aligned adaptation, not an exact reproduction**: the supplied dataset does not contain the paper's original 13 curated playlists or the complete 2,095 artist-genre one-hot features used in the paper.

## 2. What the project does

1. Loads and validates the Spotify songs CSV.
2. Selects 13 playlists deterministically using random seed `42` from playlists with at least 100 track rows.
3. Removes rows missing track title/artist, tracks appearing in multiple selected playlists, and duplicate track IDs.
4. Explores playlist class counts, energy distribution and numerical feature correlations.
5. Creates two feature representations:
   - **Paper-inspired audio features (10):** danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence and tempo.
   - **Extended representation (31 transformed features):** 19 numeric inputs including audio attributes, duration/log-duration, popularity, release-year features and engineered interactions, plus 12 one-hot musical-key columns.
6. Compares Perceptron, SVM kernels, PyTorch neural networks, Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, an MLP and soft-voting ensembles.
7. Uses five-fold `StratifiedGroupKFold` cross-validation grouped by artist, then reports results on a held-out split with zero artist overlap.
8. Exports metrics, plots and trained models into `artifacts/`.
9. Includes a script for top-three playlist predictions from a manually specified song feature vector.

## 3. Repository contents

| Path | Description |
|---|---|
| `spotify_playlist_curator_paper_aligned_combined_executed.ipynb` | Main end-to-end notebook, with data preparation, EDA, feature engineering, training, model comparison and export. |
| `spotify_songs.csv` | Input CSV dataset (32,833 rows x 23 columns). |
| `22-1.pdf` | Reference paper, *Training a Playlist Curator Based on User Taste*. |
| `predict_custom_song.py` | Terminal-based prediction demo using the saved HistGradientBoosting model. |
| `requirements_spotify_playlist_curator.txt` | Python dependencies. |
| `artifacts/` | Saved models, metrics, classification report, confusion matrix, ablation outputs, plots and run summary. |
| `Mini_Project_Writeup.pdf` | Project write-up for submission. |
| `Mini_Project_Writeup_Editable.docx` | Editable version of the write-up. |
| `Mini_Project_Presentation.pptx` | Review presentation slides. |
| `Live_Demo_Guide.md` | Demo flow and command reminders. |
| `Submission_Checklist.md` | Deliverable checklist. |

The detailed project explanation and viva guide is included as `Spotify_Playlist_Curator_Project_Explanation.pdf` in the complete deliverables bundle.

## 4. Requirements

- Python 3.11 recommended for the supplied Windows/VS Code run instructions.
- Jupyter support (for the `.ipynb` notebook).
- Python packages listed in `requirements_spotify_playlist_curator.txt`: NumPy, Pandas, scikit-learn, SciPy, Matplotlib, joblib, PyTorch and Jupyter.
- Enough memory/CPU for grouped cross-validation and randomized hyperparameter search. The full notebook can take several minutes or longer depending on the machine.

## 5. Setup on Windows (PowerShell)

Run from the project root, the directory containing the notebook, dataset and requirements file.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_spotify_playlist_curator.txt
```

Open the notebook in VS Code, ensure the Jupyter extension is installed, and choose `.venv` as the notebook kernel.

## 6. Run the notebook

Open:

```text
spotify_playlist_curator_paper_aligned_combined_executed.ipynb
```

Select the `.venv` kernel and choose **Run All**. Running the complete notebook repeats the training, cross-validation and model searches. The notebook reads `spotify_songs.csv` from the project root (or `data/spotify_songs.csv` if present) and saves experiment outputs to `artifacts/`.

To reproduce the saved outputs, run all cells in order. A notebook that only displays previously saved outputs has not necessarily rerun the training in the current environment.

## 7. Run the custom-song demo

The inference script expects both the input CSV and the exported model. The supplied bundle contains both.

```powershell
.\.venv\Scripts\python.exe .\predict_custom_song.py
```

The script loads `artifacts/champion_histgradientboosting.joblib`, reconstructs the 13-playlist ID-to-name mapping from `spotify_songs.csv`, builds the engineered feature row, and prints the top three predicted playlists.

To try another input, edit the `CUSTOM_SONG` dictionary near the top of `predict_custom_song.py`. Required values are:

- `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`
- `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- `duration_ms`, `track_popularity`, `release_year`

Values in the current dictionary are illustrative. They are not retrieved from Spotify and should not be described as an actual measured song unless replaced with verified track features.

### If the model file is missing

Run the notebook through the final export cell, then retry the script. The expected model path is:

```text
artifacts/champion_histgradientboosting.joblib
```

## 8. Dataset preparation and evaluation design

- Original CSV: **32,833 rows x 23 columns**.
- Eligible playlists: **53** with at least 100 track rows.
- Selected classes: **13**, selected with seed `42` without replacement.
- Cleaning: missing titles/artists are removed; tracks in multiple selected playlists are removed; remaining duplicate track IDs are removed.
- Clean dataset: **1,241 rows**.
- Fixed artist-grouped split: **993 training rows / 248 test rows**.
- Artist overlap between train and test: **0**.
- Cross-validation: five-fold `StratifiedGroupKFold` with artist groups.

Artist grouping is used so the same artist is not represented in both training and testing partitions, reducing the risk of an overly optimistic result due to artist-specific patterns. The target is one of 13 playlist classes. Removing multi-playlist tracks is a single-label simplification; real playlist membership can be multi-label.

## 9. Models included

### Paper-inspired baselines on 10 audio features

- Perceptron (scaled and unscaled)
- SVM with RBF, sigmoid and polynomial kernels (scaled and unscaled)
- Two paper-inspired PyTorch networks: one sigmoid hidden layer, and two hidden transformations

### Extended-feature models on 31 transformed columns

- HistGradientBoosting
- Random Forest
- Extra Trees
- RBF SVM
- Logistic Regression
- Multi-Layer Perceptron (MLP)
- Equal and weighted soft-voting ensembles combining HistGradientBoosting, Random Forest and Extra Trees

HistGradientBoosting and Random Forest are tuned with randomized search using grouped cross-validation. Soft voting combines class-probability estimates from the three tree models.

## 10. Recorded results

The following values are from the bundled `artifacts/combined_test_model_comparison.csv`, based on one fixed held-out split:

| Model | Feature set | Test accuracy | Macro-F1 |
|---|---|---:|---:|
| Equal soft-voting ensemble | Extended 31 | **77.82%** | **0.7606** |
| Weighted soft-voting ensemble | Extended 31 | 77.42% | 0.7571 |
| Tuned HistGradientBoosting | Extended 31 | 76.61% | 0.7479 |
| Tuned Random Forest | Extended 31 | 76.21% | 0.7545 |
| Scaled RBF SVM baseline | Paper audio 10 | 42.34% | 0.4075 |
| Scaled sigmoid SVM baseline | Paper audio 10 | 37.50% | 0.3492 |
| Scaled polynomial SVM baseline | Paper audio 10 | 35.89% | 0.3352 |

### Feature-ablation result

For HistGradientBoosting, the saved five-fold artist-grouped CV mean accuracies are:

| Feature set | Mean CV accuracy | Mean CV macro-F1 |
|---|---:|---:|
| Companion-repository ORIGINAL9 | 46.82% | 0.4510 |
| Paper audio 10 | 46.02% | 0.4419 |
| Extended 31 | **73.11%** | **0.7215** |

This indicates that the extended feature representation performs better in these folds. It does not prove that every added feature independently causes the improvement. CV scores and test scores are different quantities and should not be interchanged.

## 11. Important model-selection / export note

**The current notebook has a model-consistency issue that should be resolved before making a single-model “champion” claim.**

- The Section 8 cross-validation leaderboard selects **Weighted soft-voting ensemble** (74.41% grouped-CV accuracy; macro-F1 0.7330).
- The recorded held-out comparison shows **Equal soft voting** with the highest test accuracy at 77.82%, followed by weighted soft voting at 77.42%.
- The later detailed per-class report and confusion matrix explicitly use **Tuned HistGradientBoosting** (76.61% test accuracy; macro-F1 0.7479).
- The final export cell saves `best_hgb` as `champion_histgradientboosting.joblib`.
- `predict_custom_song.py` loads that HistGradientBoosting artifact, not the weighted or equal voting ensemble.

These results are not contradictory measurements of the same model: they refer to different model-selection stages/models. To make the end-to-end project consistent, choose a model based on training-only grouped CV, evaluate that chosen model on the held-out test set, generate that model's per-class report/confusion matrix, export the same fitted model and point the prediction script to that artifact. After changing the notebook, rerun the notebook and regenerate the saved outputs. Until that is done, accurately say that the custom prediction script uses HistGradientBoosting and report each model score with its model name.

Also note a stale label in the notebook: the enhanced MLP benchmark label `MLP (128, 64)` does not match its implementation (`hidden_layer_sizes=(128)`, one hidden layer of 128 units). The ensemble code combines HistGradientBoosting, Random Forest and Extra Trees; the tuning cells tune HistGradientBoosting and Random Forest.

## 12. Understanding the metrics

- **Accuracy:** Number of correct predictions divided by all predictions.
- **Precision:** Of the examples assigned to a class, the fraction that truly belong to it.
- **Recall:** Of the examples belonging to a class, the fraction correctly found.
- **F1-score:** Harmonic mean of precision and recall.
- **Macro-F1:** Average per-class F1, weighting all 13 classes equally.
- **Confusion matrix:** Rows are true classes and columns are predicted classes; off-diagonal values reveal which playlists are confused.

The 77.82% score means about 193 of 248 held-out songs were predicted correctly for that particular equal-voting test result. It does not guarantee the same accuracy on future songs.

## 13. Reproduction boundary and limitations

The original paper reported 82% test accuracy for a one-hidden-layer neural network on its own curated 13-playlist dataset (1,044 examples and 2,105 total features: ten audio features plus 2,095 artist-genre indicators). The CSV supplied here does not include those original playlists/features. Therefore, the published 82% is a historical reference, not a directly comparable benchmark for this adaptation.

Other limitations:

- Only 13 selected playlist classes are modelled.
- Multi-playlist tracks are removed, simplifying playlist membership to one label.
- Predicted class probabilities are model estimates, not guaranteed or necessarily calibrated user-preference probabilities.
- Playlist labels and musical patterns may shift over time and across users.
- There is no live Spotify API integration or user history in the current demo.

Potential future work includes a multi-label formulation, wider/user-specific playlists, richer artist/genre features, multiple-seed evaluation, probability calibration, and a small interface for input and playlist predictions.

## 14. Presentation and live demo tips

Suggested three-minute sequence:

1. Show dataset verification and the cleaning summary.
2. Explain the EDA plots and audio-only versus extended features.
3. Show the model benchmark and feature-ablation results.
4. Explain grouped validation, test accuracy, macro-F1 and the confusion matrix.
5. Run `predict_custom_song.py` and describe its top-three output as an illustrative classification.

Be prepared to explain why scaling matters for SVMs, why grouping by artist reduces leakage, what macro-F1 adds beyond accuracy, why the extended features help in this experiment, and the distinction between a paper-aligned adaptation and an exact reproduction.

## 15. Assignment submission checklist

- [ ] Put the source code, notebook, data needed to run the project, artifacts, README and requirements in the private GitHub repository.
- [ ] Add team names and any required course details to the write-up and slides.
- [ ] Share the private repository with faculty and assigned TAs.
- [ ] Include the required PDF write-up.
- [ ] Prepare the presentation deck and working live demo.
- [ ] Verify all commands from a clean environment and resolve the model-selection/export mismatch noted above.

---

**Research reference**

Awadelkarim, A. and Coelho, K. (2018). *Training a Playlist Curator Based on User Taste*. The project notebook is inspired by this work, but uses a different supplied dataset and a reduced/different feature representation.
