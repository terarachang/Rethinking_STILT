function train {
	bs=$1
	gd_step=$2
	lr=$3
	model=$4
	ckpt=$5
	dir=$6
	tasks=$7
	data_dir=$8
	max_seq_len=$9
	seed=${10}

	cmd=(python3 run.py \
		--model_type ${model} \
		--task_name ${tasks} \
		--model_name_or_path ${ckpt} \
		--do_train \
		--data_dir ${data_dir} \
		--learning_rate ${lr} \
		--num_train_epochs 2 \
		--logging_steps 20 \
		--max_seq_length ${max_seq_len} \
		--output_dir ${dir} \
		--per_gpu_train_batch_size ${bs} \
		--gradient_accumulation_steps ${gd_step} \
		--seed ${seed}
		--overwrite_output
	)
	CUDA_VISIBLE_DEVICES=0 ${cmd[@]}
}

bs_gpu=16
model="roberta"
ckpt="roberta-large"
out_dir="roberta_gpt2"
tasks="cont"
data_dir="../datasets_m"
max_len=60
gd=1
lr=1e-5

train ${bs_gpu} ${gd} ${lr} ${model} ${ckpt} ${out_dir} ${tasks} ${data_dir} ${max_len} 42
#bash eval.sh ${gd} ${lr} ${bs_gpu} ${model} ${out_dir} ${tasks} ${data_dir} ${max_len} 0.1
