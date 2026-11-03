# PhishLens: Phishing Website Detection

PhishLens is a compact Python ML project for a machine-learning course. It compares Logistic Regression with Random Forest, picks an alert threshold using validation data, evaluates on held-out domains, explains predictions with SHAP, and provides a small URL review app.

## Dataset

The attached project proposal suggests the much larger UCI PhiUSIIL set. To meet the request for a small Kaggle dataset and support an app that accepts a URL, this project uses a 3,000-row sample from Kaggle's [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls). The bundled CSV is published on [Hugging Face](https://huggingface.co/datasets/shawhin/phishing-site-classification), whose dataset card identifies Kaggle as its source and lists ODbL metadata. After duplicate removal, the project uses 2,998 URLs: 1,500 legitimate and 1,498 phishing. Treat the resulting performance as a classroom demonstration, not a real-world benchmark.

The available records contain URLs and labels, not page contents. PhishLens extracts lexical URL features (length, punctuation, hostname structure, suspicious terms and related signals). It does not use page-title or webpage features from the proposal. It never opens websites or submits credentials.

Labels: `0` = legitimate; `1` = phishing.

## Run in VS Code

Use Python 3.9 or newer.

1. Open this folder in VS Code.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

3. Install packages: `pip install -r requirements.txt`
4. Launch the included trained prototype: `streamlit run app.py`
5. To retrain and regenerate evaluation results: `python train_model.py`

In VS Code, open **Terminal > Run Task...** and select **PhishLens: Run app** or **PhishLens: Train and evaluate**. The task definitions are saved in `.vscode/tasks.json`.

The prepared dataset is included at `data/processed/phishing_small.csv`, and a trained model is included so you can start the app after installing packages. Training writes a replacement `model.joblib` and `results.json` into this project folder. The included `prepare_data.py` can create another small sample from a full Kaggle CSV placed at `data/raw/phishing_site_urls.csv`.

The included reproducible run selected Random Forest. On the held-out domain split it reached 79.1% accuracy, 72.3% precision, 85.0% recall and 78.1% F1. These are demonstration results on this balanced 3,000-row subset; they should not be treated as expected performance on current live websites. The full run details are in `results.json`.

## Project files

- `data/processed/phishing_small.csv`: bundled, compact Kaggle-derived sample.
- `prepare_data.py`: cleans a downloaded Kaggle CSV and writes a stratified sample of at most 15,000 URLs.
- `url_features.py`: extracts consistent URL features for training and review.
- `train_model.py`: creates domain-separated train/validation/test splits, compares both classifiers, selects a threshold on validation F1, evaluates once on held-out domains, and saves the best model.
- `app.py`: accepts one URL and shows the model score, alert decision and leading SHAP feature contributions.
- `results.json`: generated evaluation summary.

## How to read the result

URLs sharing a registrable domain stay in the same split to reduce memorization. The threshold and selected model use validation data; final metrics come from the held-out test set. A legitimate prediction does not prove a website is safe. URL-only features cannot assess page content or current site behavior, and phishing tactics change over time.
