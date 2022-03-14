# ClickbaitDetector

A clunky and naïve Tensorflow implementation of a clickbait classifier.

## Setup

1. Clone this repo with `git clone https://github.com/obfuscatedgenerated/ClickbaitDetector.git`

2. OPTIONAL (recommended) - Create and activate a venv with `python -m venv env` then `./env/Scripts/Activate`

3. Install the requirements with `pip install -r requirements.txt`

## Usage

### main.py

Used for setup and training. You can adjust the epoch count by modifying the EPOCHS constant.

### quicksave.py

Deprecated due to KeyboardInterrupt catching. Used to load checkpoints and then fully save the model in the case that you abandon training early.

### generate.py

Used to predict the chance of a headline being clickbait using loaded training data.

### plot.py

Used to plot the model as a diagram. You'll need to [install GraphViz](https://graphviz.org/download/) to your machine and add it to the PATH.

### youtube.py

Used to scrape titles of a YouTube channel by ID to test for clickbait. Needs tokens.json to be configured.
