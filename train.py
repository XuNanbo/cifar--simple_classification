"""训练脚本：生成多张训练过程图片，满足报告模板要求"""
import argparse, os, random, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils import load_cifar10, CIFAR10Dataset
from model import SimpleCNN

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval(); correct, loss_sum = 0, 0.
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
    return loss_sum/len(loader.dataset), correct/len(loader.dataset)

def plot_curve(values, ylabel, out_path):
    plt.figure(); plt.plot(values)
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main(args):
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    os.makedirs(args.out_dir, exist_ok=True)

    train_imgs, train_lbls, test_imgs, test_lbls = load_cifar10(args.data_dir)
    train_loader = DataLoader(CIFAR10Dataset(train_imgs, train_lbls),
                              batch_size=args.bs, shuffle=True, num_workers=2)
    test_loader  = DataLoader(CIFAR10Dataset(test_imgs,  test_lbls),
                              batch_size=args.bs, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    tr_losses, tr_accs, te_losses, te_accs = [], [], [], []
    best_acc = 0.

    for epoch in range(1, args.epochs+1):
        model.train(); running_loss, correct = 0.,0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); logits = model(x)
            loss = loss_fn(logits, y); loss.backward(); opt.step()
            running_loss += loss.item()*x.size(0)
            correct += (logits.argmax(1)==y).sum().item()

        tr_loss = running_loss/len(train_loader.dataset)
        tr_acc  = correct/len(train_loader.dataset)
        te_loss, te_acc = evaluate(model, test_loader, device)

        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        te_losses.append(te_loss); te_accs.append(te_acc)

        print(f'Epoch {epoch:02d}: '
              f'Train Loss {tr_loss:.3f} Acc {tr_acc*100:.2f}% | '
              f'Test Loss {te_loss:.3f} Acc {te_acc*100:.2f}%')

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir,'best_model.pth'))

    # 生成多张训练过程图片
    plot_curve(tr_losses, 'Train Loss', os.path.join(args.out_dir,'train_loss.png'))
    plot_curve(te_losses, 'Test Loss', os.path.join(args.out_dir,'test_loss.png'))
    plot_curve(tr_accs, 'Train Accuracy', os.path.join(args.out_dir,'train_acc.png'))
    plot_curve(te_accs, 'Test Accuracy', os.path.join(args.out_dir,'test_acc.png'))
    print(f'Best Test Accuracy: {best_acc*100:.2f}%')

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='cifar-10-batches-py')
    ap.add_argument('--out_dir', default='outputs', type=str)
    ap.add_argument('--epochs', default=30, type=int)
    ap.add_argument('--bs', default=128, type=int)
    ap.add_argument('--lr', default=1e-3, type=float)
    args = ap.parse_args(); main(args)
