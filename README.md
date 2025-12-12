# Code-Tag-Classification

This project aims to build a machine-learning system capable of predicting algorithmic tags (e.g., math, graphs, strings, number theory, etc.) for programming exercises from the Codeforces platform.

Each dataset entry contains:

A problem description

A reference Python solution

A set of tags associated with the problem

Additional metadata (difficulty, limits, samples…)

The prediction task is multi-label classification, as each exercise can belong to several tags simultaneously.


Project Structure

Code-Tag-Classification/  
    src/
        __init__.py 
        data_loader.py   # data loader
        preprocessing_py # preprocessing the data for both model
        baseline.py      # inverse frequency model for baseline performances
        evaluation.py    # implement evaluation metrics
        model            # Bert fine tuned model
        config           # Parameters

    
    data/                    
        code_classification_dataset.zip # Compressed dataset (ZIP)

    models/
        Stock model coeffs

    predict.cli.py # module to implement CLI 

    README.md