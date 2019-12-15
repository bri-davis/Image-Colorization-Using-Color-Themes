#!/bin/sh

cd ~/Image-Colorization-Using-Color-Themes
module load singularity
CONTAINER=tensorflow-torch-fixed.simg
sing="singularity exec --nv $CONTAINER"

# Hyperparameters for training
batch_size=128
epochs=30
r_parameters=(10 25 50 75 100)
color_parameters=(1 10 50 75 100)

# A model trains/tests on a single GPU
# Training a model takes ~20 minutes
# Testing an image is <1 second
gpu=0
CUDA_EXPORT_VISIBLE_DEVICES=$gpu

# Directory structure
checkpoints_dir="./checkpoints/"
pretrained_cifar_model="${checkpoints_dir}cifar10/"
output_dir="${checkpoints_dir}terminal_outputs/"
mkdir -p $output_dir


# ___________________
#/ Cool Color Scheme \
#\___________________/
#
color="cool"
color_scheme=(65.04 -25.06 -29.04) # cool shade of blue

for r_parameter in ${r_parameters[@]}
do
	for color_parameter in ${color_parameters[@]}
	do
		l1_weight=$r_parameter
		color_weight=$color_parameter
		name="r${l1_weight}c${color_weight}"
		model_name="${checkpoints_dir}${name}"
		output_name="${checkpoints_dir}cooleval.txt"
		cp -r $pretrained_cifar10_model $model_name
		echo "Now training $model_name" &>> $output_name
		
		# Train model with selected parameters
		$sing python train.py --l1-weight $l1_weight --color-weight $color_weight --scheme-color ${color_scheme[@]} --checkpoints-path $model_name --lr-decay True --batch-size $batch_size --epochs $epochs --gpu-ids $gpu
		
		# Output validation results (colorized images) for trained model
		$sing python test2.py --checkpoints-path $model_name --batch-size $batch_size --gpu-ids $gpu
		
		# Evaluate color on colorized images
		$sing python evaluate_color.py --l1-weight $l1_weight --color-weight $color_weight --scheme-color ${color_scheme[@]} --checkpoints-path $model_name --batch-size $batch_size --epochs $epochs --gpu-ids $gpu &>> $output_name
	done
done

# ___________________ 
#/ Warm Color Scheme \
#\___________________/
#
color="warm"
color_scheme=(55.46 60.3 46.52) # warm shade of orange

for r_parameter in ${r_parameters[@]}
do
        for color_parameter in ${color_parameters[@]}
        do
                l1_weight=$r_parameter
                color_weight=$color_parameter
                name="r${l1_weight}c${color_weight}${color}"
                model_name="${checkpoints_dir}${name}"
                output_name="${checkpoints_dir}warmeval.txt"
                cp -r $pretrained_cifar10_model $model_name
                echo "Now training $model_name" &>> $output_name
		
		# Train model with selected parameters
                $sing python train.py --l1-weight $l1_weight --color-weight $color_weight --scheme-color ${color_scheme[@]} --checkpoints-path $model_name --lr-decay True --batch-size $batch_size ---
epochs $epochs --gpu-ids $gpu

                # Output validation results (colorized images) for trained model
                $sing python test2.py --checkpoints-path $model_name --batch-size $batch_size --gpu-ids $gpu

                # Evaluate color on colorized images
                $sing python evaluate_color.py --l1-weight $l1_weight --color-weight $color_weight --scheme-color ${color_scheme[@]} --checkpoints-path $model_name --batch-size $batch_size --epochs
$epochs --gpu-ids $gpu &>> $output_name

        done
done
