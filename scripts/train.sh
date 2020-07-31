# train vae first
vaepath="./model/vae.pt"
labels_per_class="1000"
if [ ! -f "$vaepath" ]; then
python3.7 main.py --train --architecture vae --epochs 20 --output $vaepath &>> log_vae.txt &
else
echo "Detected pretrained vae model"
fi
# train all other model
python3.7 main.py --train --architecture cvae --epochs 20 --output ./model/"cvae_${labels_per_class}.pt" --label --labels-per-class $labels_per_class &>> log_cvae.txt
# pretrained vae path is pre-defined so we don't need to input it
python3.7 main.py --train --architecture stackedvae --epochs 20 --output ./model/"stackedvae_${labels_per_class}.pt" --label --labels-per-class $labels_per_class &>> log_stackedvae.txt
python3.7 main.py --train --architecture gmvae --epochs 20 --output ./model/"gmvae_${labels_per_class}.pt" --label --labels-per-class $labels_per_class &>> log_gmvae.txt
