# Embedding-based query-by-example search

This is the code base for [Acoustic span embeddings for multilingual query-by-example search](https://arxiv.org/pdf/2011.11807.pdf) that will appear in SLT 2021.

```
@article{hu2020acoustic,
  title={Acoustic span embeddings for multilingual query-by-example search},
  author={Hu, Yushi and Settle, Shane and Livescu, Karen},
  journal={arXiv preprint arXiv:2011.11807},
  year={2020}
}
```

### Dependencies
python 3.7, pytorch 1.4, h5py 2.8.0, numpy, scipy, tensorboard 1.14.0

The conda environment is provided in `pt1.4.yml`

### Data
Download the processed QUESST 2015 QbE task data from the link [https://drive.google.com/file/d/1bA5NFm2joZGqmHzpiziZFD2IkhDi81BQ/view?usp=sharing](https://drive.google.com/file/d/1bA5NFm2joZGqmHzpiziZFD2IkhDi81BQ/view?usp=sharing). Unzip and put the `quesst2015` folder in the directory to run the code.

### Quick Start
There are two versions of acoustic span embeddings (ASE-concat and ASE-mean). The trained checkpoints are in `expts` folder

Run query-by-example search with ASE (mean) on QUESST 2015 dev set. Results will be stored in `expts/span-mean/2015results/`
```
cd code

# Fast run. About 1s per query. Requires 32GB RAM. maxTWV 0.238  minCnxe 0.711
python qbe_main_2015.py --dir ../expts/span-mean/ --mode mean --inc 20

# Result in paper. About 5s per query. Requires 64GB RAM. maxTWV 0.255  minCnxe 0.706
python qbe_main_2015.py --dir ../expts/span-mean/ --mode mean --inc 5
```

Run query-by-example search with ASE (concat) on QUESST 2015 dev set. Results will be stored in `expts/span-concat/2015results/`
```
cd code

# Fast run. About 1s per query. Requires 32GB RAM. maxTWV 0.171  minCnxe 0.779
python qbe_main_2015.py --dir ../expts/span-concat/ --mode concat --inc 20

# Result in paper. About 5s per query. Requires 64GB RAM. maxTWV 0.193  minCnxe 0.753
python qbe_main_2015.py --dir ../expts/span-concat/ --mode concat --inc 5
```
To get the best result in paper, fusion the score output from the ASE-mean and ASE-concat systems.

Best result on dev: maxTWV 0.323 ,  minCnxe 0.670
