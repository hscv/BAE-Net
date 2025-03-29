This code is just used for training.

1. prepare hyperspectral data, which is arranged: 
			/data/xxx/dataset/whisper35/train/
											|-- automobile
												|-- HSI
													|-- 0001.png
													|-- 0002.png
													|-- ....
													|-- ....
													|-- 0401.png
													|-- groundtruth_rect.txt
											|-- automobile10
												|-- HSI
													|-- 0001.png
													|-- 0002.png
													|-- ....
													|-- ....
													|-- 0459.png
													|-- groundtruth_rect.txt
											|-- ....
											|-- toy2
												|-- HSI
													|-- 0001.png
													|-- 0002.png
													|-- ....
													|-- ....
													|-- 0401.png
													|-- groundtruth_rect.txt

where groundtruth_rect.txt consists of: 
							'X1\tY1\tW1\tH1\t'  
							'X2\tY2\tW2\tH2\t'
							'X3\tY3\tW3\tH3\t'
							..............
((X,Y) represents the coordinates of the upper left point, (W,H) represents the width and height.)


2. train method
python pretrain/train_mdnet.py --dataset /data/xxx/dataset/whisper35/train/

where
dataset is the root path of source hyperspectral data.

The model will be saved in './models/' folder.