設定保存
conda env export -n tf-metal > environment.yml


再現
conda create --name tf-metal2 --clone tf-metal
