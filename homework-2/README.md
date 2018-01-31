# Homework 2
The project is built in an object-oriented manner. There are 2 folders
of interest: `components` and `models`. 

## Components
The `components` folder holds general functions used across
multiple models:
* The `Helper` package holds all logic for loading the index and
computing the inverted index and other statistics. This is the code
provided in the original assignment notebook.
* The `PreProcessing` package hold logic for pre-processing the
document collection to only the top 1000 documents per query
as obtained through `TF-IDF`. This is used to reduce the collection
for tractability of other methods, such as `PLM` and `GLM`.

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
 