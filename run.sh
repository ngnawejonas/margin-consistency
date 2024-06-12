#!/bin/bash

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <option>"
    exit 1
fi

# Read the value of the "option" argument
gpu="$1"


case "$gpu" in
    0)
        echo "running on gpu 0"
        models="Addepalli2021Towards_RN18 Addepalli2022Efficient_RN18 Cui2023Decoupled_WRN-28-10 Ding2020MMA Engstrom2019Robustness Hendrycks2019Using Pang2022Robustness_WRN28_10"
        ;;
    1)
        echo "running on gpu 1"
        models="Rebuffi2021Fixing_28_10_cutmix_ddpm ResNet18 ResNet50 Sehwag2021Proxy_R18 Standard Wang2023Better_WRN-28-10"
        ;;
    2)
        echo "running on gpu 2"
        models="cifar10_resnet20 cifar10_resnet32 cifar10_resnet44 cifar10_resnet56 Xu2023Exploring_WRN-28-10 Zhang2019Theoretically"
        ;;
    *)
        echo "Invalid option. Please choose 0, 1, or 2."
        ;;
esac



# models18c100="Addepalli2022Efficient_RN18"
# modelsPr18="Rade2021Helper_R18_ddpm Rebuffi2021Fixing_R18_ddpm"
# models2810="Cui2023Decoupled_WRN-28-10 Rebuffi2021Fixing_28_10_cutmix_ddpm Pang2022Robustness_WRN28_10 Hendrycks2019Using"
# models2810c100="Cui2023Decoupled_WRN-28-10 Wang2023Better_WRN-28-10 Rebuffi2021Fixing_28_10_cutmix_ddpm Pang2022Robustness_WRN28_10 Hendrycks2019Using"
# models3410="Zhang2019Theoretically Addepalli2022Efficient_WRN_34_10 Cui2023Decoupled_WRN-34-10 Sehwag2021Proxy Jia2022LAS-AT_34_10 Chen2021LTD_WRN34_10 Addepalli2021Towards_WRN34 Wu2020Adversarial Chen2020Efficient Sitawarin2020Improving"

# models="Addepalli2021Towards_PARN18 Rice2020Overfitting"

dataset="cifar10"
for model_name in $models
do 
    echo $model_name
    CUDA_VISIBLE_DEVICES=$gpu python eval.py --model-name $model_name --dataset $dataset
    echo " ----------------------------- "
done