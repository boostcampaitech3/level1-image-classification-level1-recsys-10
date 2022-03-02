python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label age --name effnet_age --augmentation TestAugmentation --data_mix cutmix --mixp 0.5 --k 0
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label mask --name effnet_mask --augmentation TestAugmentation --data_mix cutmix --mixp 0.4 --k 0
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label gender --name effnet_gender --augmentation TestAugmentation --data_mix cutmix --mixp 0.3 --k 0

python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label age --name effnet_age --augmentation TestAugmentation --data_mix cutmix --mixp 0.5 --k 1
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label mask --name effnet_mask --augmentation TestAugmentation --data_mix cutmix --mixp 0.4 --k 1
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label gender --name effnet_gender --augmentation TestAugmentation --data_mix cutmix --mixp 0.3 --k 1

python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label age --name effnet_age --augmentation TestAugmentation --data_mix cutmix --mixp 0.5 --k 2
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label mask --name effnet_mask --augmentation TestAugmentation --data_mix cutmix --mixp 0.4 --k 2
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label gender --name effnet_gender --augmentation TestAugmentation --data_mix cutmix --mixp 0.3 --k 2

python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label age --name effnet_age --augmentation TestAugmentation --data_mix cutmix --mixp 0.5 --k 3
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label mask --name effnet_mask --augmentation TestAugmentation --data_mix cutmix --mixp 0.4 --k 3
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label gender --name effnet_gender --augmentation TestAugmentation --data_mix cutmix --mixp 0.3 --k 3

python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label age --name effnet_age --augmentation TestAugmentation --data_mix cutmix --mixp 0.5 --k 4
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label mask --name effnet_mask --augmentation TestAugmentation --data_mix cutmix --mixp 0.4 --k 4
python train.py --model Effnet --epochs 5 --images fdimages --dataset MultiLabelDataset --lr 1e-4 --multi_label gender --name effnet_gender --augmentation TestAugmentation --data_mix cutmix --mixp 0.3 --k 4