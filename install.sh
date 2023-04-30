##!/bin/bash
echo "export PATH=/home/hchang/.local/bin:$PATH" >> ~/.bashrc
sudo apt install -y ffmpeg
git clone https://github.com/team-sonny/utils.git
pip install -r requirements.txt
