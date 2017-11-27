# deep-fashion-retrieval
Simple image retrieval algorithm on [deep-fashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) with pytorch

### Dependencies
- Python (Compatible to 2 and 3)
- [Pytorch](http://pytorch.org/)
- Torchvision
- PIL

[Anaconda](https://www.anaconda.com/download/) is recommended.

### Training
1. Download dataset from [DeepFashion: Attribute Prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
2. Unzip all files and set `DATASET_BASE` in `config.py`
3. Run `train.py`

The models will be saved to `DATASET_BASE/models`

### Generating feature databases
- Feature extraction
    - Set `DUMPED_MODEL` in `config.py` as trained model
    - Run `scripts/feature_extractor.py`
    
    The feature will be saved to `DATASET_BASE/features`
- Accelerating querying by clustering

### Query with a picture
Unfinished