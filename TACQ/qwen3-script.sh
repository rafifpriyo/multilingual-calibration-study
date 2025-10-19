#!/bin/bash
echo "Script Starting..."
set -x

importances_dir="./importances_dir/results"
checkpoints_dir="./model_checkpoints"

model_name="Qwen3-1.7B" 
loadstring="Qwen"
groupsize=128
datasets=("flores_Chinese") 
serial_numbers=(1234)
selector_types=("sample_abs_weight_prod_contrastive")
quantization_types=("q4" "q8")
ranking_types=("top_p_sparse") 
ratios=(".0035") 

#Initiate an empty list to store all the eventual quantized model names
declare -a quantized_model_names

# Compute Intensive Section--3GPUs for 7-8B LLMs, efficiently capture importances for each weight.
for serial_number in "${serial_numbers[@]}"
do
    for cur_dataset in "${datasets[@]}"
    do
        dataset=$cur_dataset
        calibration_dataset=$cur_dataset
        for ratio in "${ratios[@]}"
        do  
            for ranking_type in "${ranking_types[@]}"
            do
                for quantization_type in "${quantization_types[@]}"
                do
                    wbits=${quantization_type%%_*}  # remove anything from first underscore onward
                    wbits=${wbits#q}               # remove the leading 'q'
                    echo "wbits: $wbits"
                    for selector_type in "${selector_types[@]}"
                    do     
                        corrupt_model_name="${model_name}+${serial_number}+${dataset}+${wbits}bit+quantized_model"
                        if [ ! -f "$checkpoints_dir/${corrupt_model_name}.pt" ]; then
                            echo "\n\nRunning gptq with run_name: ${corrupt_model_name}"
                            pixi run python -m gptq.llama \
                                $loadstring/${model_name} \
                                $dataset \
                                --true-sequential \
                                --save_in_16bits $checkpoints_dir/${corrupt_model_name}.pt \
                                --wbits $wbits \
                                --groupsize $groupsize \
                                --seed $serial_number \
                                --no-eval
                        fi

                        run_name="${model_name}+${serial_number}+${dataset}+${selector_type}+${wbits}bit+implementation_test"
                        echo "\n\nRunning measure_importances with run_name: ${run_name}"
                        pixi run python -m measure_importances \
                            --model $model_name \
                            --corrupt_model $corrupt_model_name \
                            --dataset $dataset \
                            --run_name $run_name \
                            --checkpoints_dir $checkpoints_dir \
                            --results_dir $importances_dir \
                            --selector_type $selector_type \
                            --serial_number $serial_number \
                            --save_full_gradients \
                            --save_importances_pt_path $importances_dir/$run_name/importances.pt \
                            --override_args_yaml \
                            --plot_importances

                        corrupt_model_name="${model_name}+${serial_number}+${dataset}+${wbits}bit+quantized_model"
                        run_name="${model_name}+${serial_number}+${dataset}+${selector_type}+${wbits}bit+implementation_test"
                        echo -e "\n\nRunning make_quantization_configs with run_name: ${run_name}"
                        quant_identifier="${quantization_type}+${ranking_type}+${ratio}"
                        pixi run python -m make_quantization_configs \
                            --run_name $run_name \
                            --checkpoints_dir $checkpoints_dir \
                            --results_dir $importances_dir \
                            --serial_number $serial_number \
                            --importances_pt_path $importances_dir/$run_name/importances.pt \
                            --mask_save_path $importances_dir/$run_name/important_mask_${quant_identifier}.pt \
                            --model $model_name \
                            --quantization_type $quantization_type \
                            --ranking_type $ranking_type \
                            --configs_save_path $importances_dir/$run_name/quantization_configs_${quant_identifier}.yaml \
                            --mask_fraction $ratio \
                            --proportional_total_params \
                            --force_recompute

                        quantized_model_name="${run_name}+gptq_on_${dataset}+${quant_identifier}+quantized_model"
                        echo -e "\n\nRunning gptq with run_name: ${quantized_model_name}"
                        pixi run python -m gptq.llama \
                            $loadstring/${model_name} \
                            $dataset \
                            --true-sequential \
                            --fine-wbits-yaml $importances_dir/$run_name/quantization_configs_${quant_identifier}.yaml \
                            --save_in_16bits_pretrained $importances_dir/$run_name/${quantized_model_name}.pt \
                            --no-eval \
                            --seed $serial_number \
                            --groupsize $groupsize \
                            --important_mask $importances_dir/$run_name/important_mask_${quant_identifier}.pt

                        quantized_model_names+=("$quantized_model_name")

                        pixi run python -m multilingual_huggingface_tacq \
                            --lang ${dataset} \
                            --bit ${wbits} \
                            --save_path $importances_dir/$run_name/

                    done
                done
            done
        done
    done
done
echo "Quantized model names: ${quantized_model_names[@]}"
