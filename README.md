## QCNet
QCNet: Query Context Network for salient object detection of automatic surface inspection

#### Training 
* `git clone https://github.com/sunjie97/QCNet.git`
* Download CFRP dataset and DUTS dataset 
* Generate train.txt and test.txt file as follows:
    ```
    DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00025207.jpg,DUTS-TR/DUTS-TR-Mask/ILSVRC2014_train_00025207.png
    DUTS-TR/DUTS-TR-Image/ILSVRC2012_test_00071411.jpg,DUTS-TR/DUTS-TR-Mask/ILSVRC2012_test_00071411.png
    DUTS-TR/DUTS-TR-Image/n03764736_11208.jpg,DUTS-TR/DUTS-TR-Mask/n03764736_11208.png
    ...
    ```
* Specify backbone, dataset dir in corresponding configuration file
* Start to train with `python solver.py --cfg configs/composite.yaml --phase train`


#### Testing 
* Specify `infer_checkpoint`, `infer_save_dir` and data path in cfg file
* `python solver.py --cfg configs/composite.yaml --phase test`
* We use the public open source evaluation code: https://github.com/Hanqer/Evaluate-SOD.git

#### Pretrained models, datasets and results 
* CFRP dataset: [CFRP](https://drive.google.com/file/d/1ALbeGeUWGbazhZr5vyNyuOxPGBB-j4dl/view?usp=sharing)
* Pretrained model: [QCNet_ResNet_Composite](https://drive.google.com/file/d/17x3nhrXKgMFrN1GGJYQ0FgMlQq-WENGC/view?usp=sharing) | [QCNet_VGG_Composite](https://drive.google.com/file/d/1o8WPYmYbLbRt9Im2VE5MCyYZetWYZrYR/view?usp=sharing) | [QCNet_ResNet_DUTS](https://drive.google.com/file/d/1oP2QXPLUEcXuGzB3_SQ5lFzHh5MgYP2B/view?usp=sharing) | [QCNet_VGG_DUTS](https://drive.google.com/file/d/18d4Zau-N_5Y1JZ7mgBbWZ23kHRQ0esB_/view?usp=sharing)
* Saliency maps: [QCNet_ResNet and QCNet_VGG](https://drive.google.com/file/d/1XIySjx0CNIU22NBRc0iqniqtgMOW4ge1/view?usp=sharing)