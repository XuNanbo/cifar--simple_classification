"""生成训练/测试集图像示意图"""
import argparse, os, numpy as np, matplotlib.pyplot as plt
from data_utils import load_cifar10

LABELS = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def save_grid(imgs, labels, out_path, num=6):
    idx = np.random.choice(len(imgs), num, replace=False)
    plt.figure(figsize=(12,3))
    for i, id_ in enumerate(idx,1):
        plt.subplot(1,num,i)
        img = imgs[id_].reshape(3,32,32).transpose(1,2,0)
        plt.imshow(img.astype('uint8')); plt.axis('off')
        plt.title(LABELS[labels[id_]])
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main(args):
    train_imgs, train_lbls, test_imgs, test_lbls = load_cifar10(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    save_grid(train_imgs, train_lbls, os.path.join(args.out_dir,'train_samples.png'), args.num)
    save_grid(test_imgs,  test_lbls,  os.path.join(args.out_dir,'test_samples.png'),  args.num)
    print('示意图已保存到', args.out_dir)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='cifar/cifar-10-batches-py')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--num', default=6, type=int)
    args = ap.parse_args(); main(args)
