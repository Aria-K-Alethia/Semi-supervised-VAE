# train vae first
vaepath="./model/vae.pt"
if [ ! -f "$vaepath" ]; then
python3.7 main.py --train --architecture vae --epochs 20 --output $vaepath &>> log_vae.txt &
else
echo "Detected pretrained vae model"
fi
# train all other model
python3.7 main.py --train --architecture cvae --epochs 20 --output ./model/cvae.pt --label &>> log_cvae.txt
# pretrained vae path is pre-defined so we don't need to input it
python3.7 main.py --train --architecture stackedvae --epochs 20 --output ./model/stackedvae.pt --label &>> log_stackedvae.txt
python3.7 main.py --train --architecture gmvae --epochs 20 --output ./model/gmvae.pt --label &>> log_gmvae.txt
