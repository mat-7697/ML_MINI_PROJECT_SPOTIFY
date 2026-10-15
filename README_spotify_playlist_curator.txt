SPOTIFY PLAYLIST CURATOR — COMBINED PAPER-ALIGNED NOTEBOOK

FILES
- spotify_playlist_curator_paper_aligned_combined_executed.ipynb : combined notebook with saved cell outputs
- spotify_songs.csv                                      : the supplied dataset
- 22-1.pdf                                               : the research paper, Training a Playlist Curator Based on User Taste (2018)
- artifacts/                                             : tables, plots, fitted scikit-learn models, PyTorch state_dicts, and run metadata

WHAT THE NOTEBOOK DOES
1. Loads and validates the supplied 32,833 x 23 Spotify CSV.
2. Selects 13 eligible playlists deterministically (seed 42), cleans ambiguous track memberships, and uses an artist-grouped train/test split.
3. Implements paper-aligned Perceptron and SVM baselines (RBF, sigmoid, polynomial), scaled and unscaled, using the paper's 10 listed audio features.
4. Trains paper-inspired one- and two-hidden-layer PyTorch networks using full-batch NLL training and L2 regularization for 150 epochs.
5. Benchmarks enhanced metadata/engineered features with Logistic Regression, SVC, Random Forest, Extra Trees, MLP, and HistGradientBoosting.
6. Includes an explicit ORIGINAL9 vs paper-10-audio-features vs extended-31-feature ablation, plus model tuning and a final shared test comparison.
7. Exports metrics and model artifacts into artifacts/.

IMPORTANT SCIENTIFIC LIMITATION
This is a paper-aligned adaptation, not an exact reproduction of the paper's original experiment. The paper's original 13 playlists and its 2,095 artist-genre one-hot features are not present in the supplied CSV. The paper's published 82% result is therefore shown as a reference, not a directly comparable target.

VERIFIED RESULTS FROM THE COMBINED RUN
- Paper-feature, scaled RBF SVM: about 42.1% grouped CV accuracy; about 42.3% on the held-out test set.
- Explicit feature ablation: HistGradientBoosting improves from about 46.0% CV accuracy with the paper's 10 audio features to about 73.1% with the extended 31 transformed features.
- Tuned HistGradientBoosting: about 76.6% held-out test accuracy in this fresh run.
- Equal soft-voting ensemble: about 77.8% held-out test accuracy in this fresh run.
- The originally supplied enhanced notebook had a cached run reporting 78.23% for HistGradientBoosting. Fresh scores can vary slightly across software/library versions; the combined notebook recomputes rather than hard-codes scores.

HOW TO RUN
1. Keep the notebook and spotify_songs.csv in the same folder (or put the CSV in data/).
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells from top to bottom. The full hyperparameter search may take several minutes.
4. For a faster smoke test, set QUICK = True in the setup cell; this reduces the search budget and can change final metrics.

DEPENDENCIES
Python 3.10+ recommended; numpy, pandas, scikit-learn, scipy, matplotlib, joblib, torch, and Jupyter Notebook/JupyterLab. Google Colab normally supplies most dependencies. If PyTorch is missing locally, install it using the official instructions for your OS/accelerator.

REPRODUCIBILITY NOTES
- The seed is fixed at 42.
- Train/test and CV folds are grouped by normalized artist name to prevent artists leaking between folds.
- Preprocessors live inside scikit-learn pipelines and are fit on each training fold.
- The test set is held until the final comparison section.
- The selected hyperparameter search CV score is a model-selection score, not an independent estimate; the fixed held-out test score is reported separately.
