file_in="data/raw_data/all.jsonl"
file_out="data/data_processed/squad_mrc.jsonl"

echo $file_in
echo $file_out

python modules/datasets/convert_to_mrc.py --file_in=$file_in --file_out=$file_out
python modules/datasets/train_valid_split.py --file_in=$file_out