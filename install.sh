##!/bin/bash
echo "export PATH=/home/hchang/.local/bin:$PATH" >> ~/.bashrc
sudo apt install -y ffmpeg
git clone https://github.com/team-sonny/utils.git
pip install -r requirements.txt
sudo mkdir -p ~/datadisk
sudo mount -o discard,defaults /dev/sdb ~/datadisk/
ln -s ~/datadisk/KEMDy20 .
ln -s ~/datadisk/wav_data .
source ~/.bashrc
wandb login
