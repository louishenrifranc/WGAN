# WGAN in Tensorflow
__Implementation of a [WGAN](https://arxiv.org/abs/1701.07875) for MNIST,
with checkpoint in model/__

# Requirements
* tensorflow=1.0.0
* numpy
* tqdm
* argparse

# Train
To train the model, run the command:
```
python model.py --train
```
Model retrieve the last checkpoint, and saved it every 100 iterations.
It also saved summary for images generated and other variables, to further
display in Tensorboard ```tensorboard --logdir="logs"```

# Draw
To draw sample from the generator, run the command:
```
python model.py --draw 1,2,3,4,5,7,9
```

### Samples generated
![Alt text](sample/image.png?raw=true "Title")
