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
	warm=${11}

	cmd=(python3 run.py \
		--model_type ${model} \
		--task_name ${tasks} \
		--model_name_or_path ${ckpt} \
		--do_train \
		--data_dir ${data_dir} \
		--learning_rate ${lr} \
		--num_train_epochs 5 \
		--logging_steps 100
		--max_seq_length ${max_seq_len} \
		--output_dir ${dir} \
		--per_gpu_train_batch_size ${bs} \
		--gradient_accumulation_steps ${gd_step} \
		--seed ${seed} \
		--warmup_ratio ${warm} \
		--overwrite_output
	)
	CUDA_VISIBLE_DEVICES=0 ${cmd[@]}
}

ckpt_arr=("roberta-large" "roberta_gpt2" "roberta_hella-p" "roberta_hella")
seed_arr=(12 42)
lr_arr=(5e-6 1e-5 2e-5)
wup_arr=(0 0.2)
bs_arr=(32 16 8)
model="roberta"
ckpt="roberta-large"
out_dir="cola"
tasks="cola"
data_dir="glue_data/CoLA"
max_len=50

mkdir "${tasks}_npy"


for c in "${ckpt_arr[@]}"
do
	for s in "${seed_arr[@]}"
	do
		for i in "${lr_arr[@]}"
		do
			for j in "${bs_arr[@]}"
			do
				for w in "${wup_arr[@]}"
				do
					train $j 1 $i ${model} $c ${out_dir} ${tasks} ${data_dir} ${max_len} $s $w
					bash eval.sh 1 $i $j ${model} ${out_dir} ${tasks} ${data_dir} ${max_len} $w
					echo "------------${c}: seed=${s}, BS=${j}, lr=${i}, warmup=${w} done ! ------------"
				done
			done
		done
		python3 get_all_results.py ${out_dir} mcc  | tee "${tasks}_npy/${c}_seed_${s}.txt"
		mv "${out_dir}/all_results.npy" "${tasks}_npy/${c}_seed_${s}.npy"
		rm -r ${out_dir}
	done
done

