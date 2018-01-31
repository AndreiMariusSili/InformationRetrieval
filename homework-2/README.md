# Homework 2
The project is built in an object-oriented manner. There are 2 folders
of interest: `components` and `models`. 

## Components
The `components` folder holds general functions used across
multiple models:
* The `Helper` package holds all logic for loading the index and
computing the inverted index and other statistics. This is the code
provided in the original assignment notebook.
* The `PreProcessing` package holds logic for pre-processing the
document collection to only the top 1000 documents per query
as obtained through `TF-IDF`. This is used to reduce the collection
for tractability of other methods, such as  `LDA`, `LSI`, `PLM`
and `GLM`.
* The `LTR_Process_Data` package contains code for procesing and
loading data for the LearnToRank algorithm. The two classes,
`TrainingDataLoader` and `ValidatingDataLoader` load the training
and validating data, respectively. The loaded data is eventually saved
to .pkl files for future use.
* The `LogRegression` package presents the code for the logistic 
regression model, including loading the training and validation data.
If previous data was already created, it tries to load it from .pkl
files. Otherwise, it creates new data using the classes from 
`LTR_Process_Data`. Ultimately, it predicts the rankings and saves them
to specific files.

## Models
The `models` folder holds all implemented models. Each model is 
initialized with the index, inverted index, and other statistics
depending on the needs. Optionally, a document collection can be 
passed to be used on retrieval. If it is not passed, the model
will run over all the documents and queries in the index.

`TFIDF` and `BM25` inherit from `VectorSpaceModel`. `JelinekMercer`,
`AbsoluteDiscounting` and `DirichletPrior` inherit from `LanguageModel`.
The models share quite some common features so grouping them in such
a way made sense.

`PositionalLanguageModel` and `GeneralizedLanguageModel` are both
particular in their implementation so they are left as standalone.

The `LatentSemanticModels` package contains implementations for the 
latent semantic models: Latent Semantic Indexing and Latent Dirichlet 
Allocation. The class `Sentences2Vec` is also included here since it 
contains our extended implementation of the `IndriSentences` class 
and is used in both LSMs.
 