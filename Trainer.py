
import torch
import dataset
import os
import argparse
from torch.utils.data import DataLoader
import models
import time
import matplotlib.pyplot as plt
import loss


class Trainer:
    record = {"train_loss_d": [], "train_loss_g": [], "train_psnr": [], "val_loss": [], "val_psnr": []}
    x_epoch = []

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.gnet = models.Generator()
        self.dnet = models.Discriminator()
        batch = self.args.batch
        self.train_loader = DataLoader(dataset.SRGANDataset(self.args.data_path, "train"),
                                       batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(dataset.SRGANDataset(self.args.data_path, "val"),
                                     batch_size=batch, shuffle=False, drop_last=True)
        self.criterion_g = loss.PerceptualLoss(self.device)
        self.regularization = loss.RegularizationLoss()
        self.criterion_d = torch.nn.BCELoss()
        self.epoch = 0
        self.lr = 1e-3
        self.best_psnr = 0.
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params, start training...")
            else:
                param_dict = torch.load(self.args.save_path)
                self.epoch = param_dict["epoch"]
                self.lr = param_dict["lr"]
                self.dnet.load_state_dict(param_dict["dnet_dict"])
                self.gnet.load_state_dict(param_dict["gnet_dict"])
                self.best_psnr = param_dict["best_psnr"]
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}    [best_psnr]: {}".format(self.args.save_path,
                                                                                                self.epoch, self.lr,
                                                                                                self.best_psnr))
        self.dnet.to(self.device)
        self.gnet.to(self.device)
        self.optimizer_d = torch.optim.Adam(self.dnet.parameters(), lr=self.lr)
        self.optimizer_g = torch.optim.Adam(self.gnet.parameters(), lr=self.lr*0.1)
        self.real_label = torch.ones([batch, 1, 1, 1]).to(self.device)
        self.fake_label = torch.zeros([batch, 1, 1, 1]).to(self.device)

    @staticmethod
    def calculate_psnr(img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    def train(self, epoch):
        self.dnet.train()
        self.gnet.train()
        train_loss_d = 0.
        train_loss_g = 0.
        train_loss_all_d = 0.
        train_loss_all_g = 0.
        psnr = 0.
        total = 0
        start = time.time()
        print("Start epoch: {}".format(epoch))
        for i, (img, label) in enumerate(self.train_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            fake_img = self.gnet(img)
            loss_g = self.criterion_g(fake_img, label, self.dnet(fake_img)) + 2e-8*self.regularization(fake_img)
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()
            if i % 2 == 0:
                real_out = self.dnet(label)
                fake_out = self.dnet(fake_img.detach())
                loss_d = self.criterion_d(real_out, self.real_label
                                          ) + self.criterion_d(fake_out, self.fake_label)
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                train_loss_d += loss_d.item()
                train_loss_all_d += loss_d.item()
            train_loss_g += loss_g.item()
            train_loss_all_g += loss_g.item()
            psnr += self.calculate_psnr(fake_img, label).item()
            total += 1

            if (i+1) % self.args.interval == 0:
                end = time.time()
                print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} dnet_loss:{:.5f} gnet_loss:{:.5f} psnr:{:.4f}".format(
                    epoch, (i+1)*100/len(self.train_loader), end-start,
                    train_loss_d/self.args.interval,
                    train_loss_g/self.args.interval, psnr/total
                ))
                train_loss_d = 0.
                train_loss_g = 0.
        print("Save params to {}".format(self.args.save_path1))
        param_dict = {
            "epoch": epoch,
            "lr": self.lr,
            "best_psnr": self.best_psnr,
            "dnet_dict": self.dnet.state_dict(),
            "gnet_dict": self.gnet.state_dict()
        }
        torch.save(param_dict, self.args.save_path)
        return train_loss_all_d/len(self.train_loader), train_loss_all_g/len(self.train_loader), psnr/total

    def val(self, epoch):
        self.gnet.eval()
        self.dnet.eval()
        print("Test start...")
        val_loss = 0.
        psnr = 0.
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                fake_img = self.gnet(img).clamp(0.0, 1.0)
                loss = self.criterion_g(fake_img, label, self.dnet(fake_img))
                val_loss += loss.item()
                psnr += self.calculate_psnr(fake_img, label).item()
                total += 1

            mpsnr = psnr / total
            end = time.time()
            print("Test finished!")
            print("[Epoch]: {} time:{:.2f} loss:{:.5f} psnr:{:.4f}".format(
                epoch, end - start, val_loss / len(self.val_loader), mpsnr
            ))
            if mpsnr > self.best_psnr:
                self.best_psnr = mpsnr
                print("Save params to {}".format(self.args.save_path))
                param_dict = {
                    "epoch": epoch,
                    "lr": self.lr,
                    "best_psnr": self.best_psnr,
                    "gnet_dict": self.gnet.state_dict(),
                    "dnet_dict": self.dnet.state_dict()
                }
                torch.save(param_dict, self.args.save_path1)
        return val_loss/len(self.val_loader), mpsnr

    def draw_curve(self, fig, epoch, train_loss_d, train_loss_g, train_psnr, val_loss, val_psnr):
        ax0 = fig.add_subplot(121, title="loss")
        ax1 = fig.add_subplot(122, title="psnr")
        self.record["train_loss_d"].append(train_loss_d)
        self.record["train_loss_g"].append(train_loss_g)
        self.record["train_psnr"].append(train_psnr)
        self.record["val_loss"].append(val_loss)
        self.record["val_psnr"].append(val_psnr)
        self.x_epoch.append(epoch)
        ax0.plot(self.x_epoch, self.record["train_loss_d"], "bo-", label="train_d")
        ax0.plot(self.x_epoch, self.record["train_loss_g"], "go-", label="train_g")
        ax0.plot(self.x_epoch, self.record["val_loss"], "ro-", label="val_g")
        ax1.plot(self.x_epoch, self.record["train_psnr"], "bo-", label="train")
        ax1.plot(self.x_epoch, self.record["val_psnr"], "ro-", label="val")
        if epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(r"./train_fig/train_{}.jpg".format(epoch))

    def lr_update(self):
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = self.lr * 0.1
        self.lr = self.optimizer_d.param_groups[0]["lr"]
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = self.lr
        print("===============================================")
        print("Learning rate has adjusted to {}".format(self.lr))


def main(args):
    t = Trainer(args)
    fig = plt.figure()
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss_d, train_loss_g, train_psnr = t.train(epoch)
        val_loss, val_psnr = t.val(epoch)
        t.draw_curve(fig, epoch, train_loss_d, train_loss_g, train_psnr, val_loss, val_psnr)
        # if (epoch + 1) % 10 == 0:
        #     t.lr_update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training SRGAN with celebA")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_path", default=r"T:\srgan", type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--save_path", default=r"./weight01.pt", type=str)
    parser.add_argument("--save_path1", default=r"./weight00.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=8, type=int)
    args1 = parser.parse_args()
    main(args1)

