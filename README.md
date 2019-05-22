# pytorch-mask-rcnn


This is a Pytorch implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is a fork of the
[implementation from multimodallearning](https://github.com/multimodallearning/pytorch-mask-rcnn), that in turn
is in large parts based on Matterport's [Mask_RCNN](https://github.com/matterport/Mask_RCNN). Matterport's repository is
an implementation on Keras and TensorFlow.

The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based
on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/street.png)

The next four images visualize different stages in the detection pipeline:


##### 1. Anchor sorting and filtering
The Region Proposal Network proposes bounding boxes that are likely to belong to an object. Positive and negative anchors
along with anchor box refinement are visualized.

![](assets/detection_anchors.png)


##### 2. Bounding Box Refinement
This is an example of final detection boxes (dotted lines) and the refinement applied to them (solid lines) in the second stage.

![](assets/detection_refinement.png)


##### 3. Mask Generation
Examples of generated masks. These then get scaled and placed on the image in the right location.

![](assets/detection_masks.png)


##### 4. Composing the different pieces into a final result

![](assets/detection_final.png)

## Changes from the multimodellearning implementation
* Compatible with PyTorch v1.x
* Compatible with Windows
* Can train with batch sizes > 1
* Separated FPN, FCNN and Mask RCNN components into separate modules
* Support for data augmentation, both and training and test time (not very easy to use yet)


## Requirements
* Python 3
* Pytorch 1.x
* matplotlib, scipy, skimage

## Installation
1. Clone this repository.

        git clone https://github.com/Britefury/pytorch-mask-rcnn.git

2. Switch to the `refactor-pytorch-1.0` branch.

3. Build the native C++ and CUDA extensions

        python setup.py build_ext

4. Train the ellipses example program.

a. Write the config file `smallobjects.cfg`:

```
[paths]
ellipses_root=<path for ellipses dataset>
```

b. Create the ellipses dataset

        python create_ellipses.py

c. Train a network.

For Mask R-CNN instance segmentation:

        python example_smallobject_train --dataset=ellipses --head=mask_rcnn --plot_dir=OUTPUT_PATH --prediction_every_n_epochs=10

Every 10 epochs some plots will be generated and saved to OUTPUT_PATH, visualising the network's predictions.

For Faster R-CNN object detection:

        python example_smallobject_train --dataset=ellipses --head=faster_rcnn --plot_dir=OUTPUT_PATH --prediction_every_n_epochs=10

For RPN object detection:

        python example_smallobject_train --dataset=ellipses --head=rpn --plot_dir=OUTPUT_PATH --prediction_every_n_epochs=10
