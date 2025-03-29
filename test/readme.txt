This code is just used as testing.

1. prepare hyperspectral data, which is arranged: 
			/data/xxx/dataset/whisper35/test/
											|-- test_HSI
												|-- ball
													|-- groundtruth_rect.txt
													|-- HSI
														|-- 0001.png
														|-- 0002.png
														|-- ....
														|-- ....
												|-- basketball
													|-- groundtruth_rect.txt
													|-- HSI
														|-- 0001.png
														|-- 0002.png
														|-- ....
														|-- ....
												|-- ....
												|-- worker
													|-- groundtruth_rect.txt
													|-- HSI
														|-- 0001.png
														|-- 0002.png
														|-- ....
														|-- ....


where groundtruth_rect.txt consists of: 
							'X1\tY1\tW1\tH1\t'  
							'X2\tY2\tW2\tH2\t'
							'X3\tY3\tW3\tH3\t'
							..............
((X,Y) represents the coordinates of the upper left point, (W,H) represents the width and height.)


2. test method
python tracking/run_tracker.py --seq /data/xxx/dataset/whisper35/test/test_HSI/ --savepath './results/' --channel_model 'models/epochxxx.pth'

where
seq is the root path of source hyperspectral data.
savepath is the path used for saving the tracking results.
channel_model is the path of band attention model.