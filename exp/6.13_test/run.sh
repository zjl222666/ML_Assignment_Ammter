PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p DBX32 -n1 --gres=gpu:1  --cpus-per-task=5 \
python /mnt/cache/zhengjinliang/Ammeter/main.py \
   --batch_size 6 \
   --epochs 600 \
   --resume ckpt/checkpoint.pth \
   --output_dir ckpt \
   --dim_feedforward 2048

