# GAN experiments with multiple datasets

This project is the result of several tests using GAN, with different datasets as input, all different in number of files and image complexity. Some datasets were constructed using scrapy. The technology used is Tensorflow (with Cuda enabled) and more specifically, the `tf.contrib.GANEstimator`.

Some good reads about GAN that helped me :
- https://ajolicoeur.wordpress.com/cats/
- https://medium.freecodecamp.org/how-ai-can-learn-to-generate-pictures-of-cats-ba692cb6eae4
- https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb
- https://medium.com/@jonathan_hui/gan-a-comprehensive-review-into-the-gangsters-of-gans-part-2-73233a670d19
- https://julianzaidi.wordpress.com/2017/04/24/deep-convolution-gan-dcgan-architecture-and-training/
- https://github.com/soumith/ganhacks/blob/master/README.md


Datasets that were used :
- Mnist
- Pokemon (scrapped from official site)
- Yugioh (scrapped from official site)
- Magic (scrapped from official site)

## Project files arrangement

- CreatureCrawler/ : contains spiders to construct datasets
- GANs/ : choose one subfolder
    - gen_img/ : generated data, title of each image is the epoch it was produced on
    - preprocess/ : eventual preprocessing of images using OpenCV
    - utils/ :
        - data_provider.py : read data from TFRecord and return Dataset of image after needed preprocess (resize, normalization) and noise for generator input
        - download_and_convert_*.py : convert images in their original form to unique TFRecord
        - networks.py : contains generator and discriminator models
    - *_GAN.py : init flags, get dataset, create GANEstimator and launch training
    

## Results

### Mnist
The dataset is numerous, simple and consistent so overall good, clean, stable results.

![Alt text](./GANs/mnist/example_generator_output.png?raw=true "Mnist output")

### Pokemon
The dataset is really small, data is simple but heterogenous, generator was able to copy  completely some images, but not invent, and there was lots of mode collapse. Interesting nonetheless to see it produce copy of images it has never directly seen.

![Alt text](./GANs/pokemon/example_generator_output.png?raw=true "Pokemon output")

### Yugioh
The dataset is small with complex, heterogenous data. Generation could not converge, total mode collapse and degradation with more training, only color theme evoke real data.

![Alt text](./GANs/yugioh/example_generator_output.png?raw=true "Yugioh output")

### Magic
The dataset is sufficiently large, data is complex and heterogenous. Except for the always present mode collapse problem, the generated images are quite detailed, evoking colors and shape from real ones.

![Alt text](./GANs/magic/example_generator_output.png?raw=true "Magic output")
