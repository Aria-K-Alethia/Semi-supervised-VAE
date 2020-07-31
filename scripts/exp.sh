labels_per_class=(10 100 1000)
models=("cvae" "stackedvae" "gmvae")
for nlabel in ${labels_per_class[*]}; do
    echo "========================${nlabel}=========================="
    for model in ${models[*]}; do
        echo "$model"
        python3.7 main.py --label --architecture ${model} --output ./model/"${model}_${nlabel}.pt" --labels-per-class ${nlabel}
    done
done