"""测试脚本：生成混淆矩阵与示例预测图"""
import argparse, os, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from data_utils import load_cifar10, CIFAR10Dataset
from model import SimpleCNN

LABELS = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def tensor_to_img(t):
    return (t.permute(1,2,0).numpy()*255).astype('uint8')

def main(args):
    train_imgs, train_lbls, test_imgs, test_lbls = load_cifar10(args.data_dir)
    test_ds = CIFAR10Dataset(test_imgs, test_lbls)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    preds, gts = [], []
    imgs_list = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); logits = model(x); p = logits.argmax(1).cpu().numpy()
            preds.extend(p); gts.extend(y.numpy())
            imgs_list.extend(x.cpu())

    # 混淆矩阵
    cm = confusion_matrix(gts, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(xticks_rotation='vertical'); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,'confusion_matrix.png')); plt.close()

    # 随机正确/错误示例
    preds = np.array(preds); gts = np.array(gts)
    correct_idx = np.where(preds==gts)[0]
    wrong_idx   = np.where(preds!=gts)[0]

    def save_sample(idx_list, tag):
        sel = np.random.choice(idx_list, 6, replace=False)
        plt.figure(figsize=(9,3))
        for i, idx in enumerate(sel,1):
            plt.subplot(1,6,i)
            plt.imshow(tensor_to_img(imgs_list[idx]))
            plt.title(f'{LABELS[preds[idx]]}\nGT:{LABELS[gts[idx]]}', fontsize=8)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir,f'{tag}_samples.png')); plt.close()

    save_sample(correct_idx, 'correct')
    save_sample(wrong_idx, 'wrong')
    print('测试结果图片已保存到', args.out_dir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='cifar/cifar-10-batches-py')
    ap.add_argument('--ckpt', default='outputs/best_model.pth')
    ap.add_argument('--out_dir', default='outputs')
    args = ap.parse_args(); main(args)
