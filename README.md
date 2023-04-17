# EECS-6322-NN-and-DL-project
Neural Networks and Deep Learning Winter 2023 Course Project Reproducibility Challenge : Camouflaged Object Segmentation with Distraction Mining


Abstract
The paper proposes a new approach to camouflaged object segmentation (COS) called the Positioning and Focus Network (PFNet), which is inspired by the process of predation in nature. COS aims to identify objects that blend perfectly into their surroundings, which is challenging due to the similarities between candidate objects and the noise background. The PFNet is composed of two modules: the positioning module (PM) and the focus module (FM). The PM mimics the detection process in predation to position potential target objects from a global perspective. The FM progressively refines the coarse prediction by focusing on the ambiguous regions to perform the identification process in predation. The paper introduces a novel distraction mining strategy in the FM that significantly improves the performance of estimation. The PFNet is shown to run in real-time and outperform 18 cutting-edge models on three challenging datasets under four standard metrics. The reader aims to reproduce the paper's results using the same datasets and methodologies to verify the findings, learn from their methods, and build upon them.

You can download the reproduced trained PFNet file by [Clicking here](https://drive.google.com/file/d/1AOFw2hXt3B6DTlB3txneyH6LXjNoPB52/view?usp=sharing). And put it in your directory based on the config.py file.
