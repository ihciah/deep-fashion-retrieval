# deep-fashion-retrieval
Simple image retrieval algorithm on [deep-fashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) with pytorch

![Capture](resources/Capture.PNG)

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
    - Seems no need to do this... A naive query on the database with 139,709 features cost about 0.12 sec. 

### Query with a picture
- Run `retrieval.py img_path`, for example:

    `python retrieval.py img/Sheer_Pleated-Front_Blouse/img_00000005.jpg`

