import os 
import argparse
import shutil

def copy_image(img_list, src_dir, des_dir, count = 0) : 
	
	for img in img_list : 
		src = os.path.join(src_dir, img)
		if 'jpg' in src : 
			dst = os.path.join(des_dir, '{:d}.jpg'.format(count))
		elif 'JPG' in src: 
			dst = os.path.join(des_dir, '{:d}.JPG'.format(count)) 
		shutil.copyfile(src, dst)
		count += 1

parser = argparse.ArgumentParser(description='split a directory of images into train / val / test',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--img-dir', type=str, default=None, help='input image directory')
parser.add_argument('--out-dir', type=str, default=None, help='output image directory')
parser.add_argument('--split-ratio', default=[0.7, 0.1, 0.2], nargs="+", type=float, help="split ratio (train, val and test)")


args = parser.parse_args()


os.mkdir(args.out_dir)

cls_name = sorted(os.listdir(args.img_dir))

print (cls_name)

## create training / val / test
train_dir =  os.path.join(args.out_dir, 'train')
val_dir =  os.path.join(args.out_dir, 'val')
test_dir =  os.path.join(args.out_dir, 'test')

os.mkdir(train_dir)
os.mkdir(val_dir)
os.mkdir(test_dir)

for cls in cls_name : 
	in_cls_dir = os.path.join(args.img_dir, cls)
	img_list = sorted(os.listdir(in_cls_dir)) ## sorting the images, make sure everyone has the same split
	img_list = [img for img in img_list if img[-3:] == 'jpg' or img[-3:] == 'JPG' ]
	nb_img = len(img_list)

	nb_train = int(args.split_ratio[0] * nb_img)
	nb_val = int(args.split_ratio[1] * nb_img)
	nb_test = nb_img - nb_train - nb_val

	out_cls_train_dir = os.path.join(train_dir, cls)
	os.mkdir(out_cls_train_dir)
	copy_image(img_list[ : nb_train], in_cls_dir, out_cls_train_dir, count = 0)

	out_cls_val_dir = os.path.join(val_dir, cls)
	os.mkdir(out_cls_val_dir)
	copy_image(img_list[nb_train : nb_train + nb_val], in_cls_dir, out_cls_val_dir, count = nb_train)

	out_cls_test_dir = os.path.join(test_dir, cls)
	os.mkdir(out_cls_test_dir)
	copy_image(img_list[nb_train + nb_val : ], in_cls_dir, out_cls_test_dir, count = nb_train + nb_val)

	msg = 'cls name {} : \n \t train images --> {:d}; \n \t val images --> {:d}; \n \t test images --> {:d}'.format(cls, nb_train, nb_val, nb_test)
	print (msg)


