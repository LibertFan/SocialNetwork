import os
import shutil
import time
import argparse
import torch
import numpy as np
from DataLoader import DataLoader as DataStream
from AutoRec import AutoRec as Model


def main():
    opts = read_commands()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.device
    trainer = Trainer(opts)
    if opts.is_training:
        trainer.train()
        trainer.test()
    else:
        trainer.test()


class Trainer(object):
    def __init__(self, opts):
        self._options = opts
        self.model_name = opts.model_name + '_' + opts.tag
        self.log_file = os.path.join(opts.log_dir, self.model_name+'_{}.txt'.format(
            time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))))
        self.save_folder = os.path.join(opts.save_dir, self.model_name) \
            if opts.save_folder is None else opts.save_folder
        self.util_folder = os.path.join(opts.util_dir, self.model_name) \
            if opts.util_folder is None else opts.util_folder
        if opts.is_training:
            if os.path.exists(self.log_file):
                del_cmd = input('[Warning][LogFile {} exists][Delete it?]'.format(self.log_file))
                if del_cmd:
                    os.remove(self.log_file)
            if os.path.exists(self.save_folder):
                del_cmd = bool(eval(input('[Warning][SaveFile {} exists][Delete it?]'.format(self.save_folder))))
                if del_cmd:
                    shutil.rmtree(self.save_folder)
            os.mkdir(self.save_folder)
            if os.path.exists(self.util_folder):
                del_cmd = bool(eval(input('[Warning][UtilFile {} exists][Delete it?]'.format(self.util_folder))))
                if del_cmd:
                    shutil.rmtree(self.util_folder)
            os.mkdir(self.util_folder)
        self.streamer = DataStream(opts)
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.model = Model(opts, self.streamer.train_user_dat)
        self.epoch = 0
        self.best_score = 1e10

    def train(self):
        opts, model, streamer = self._options, self.model, self.streamer
        self.adjust()
        for e in range(opts.epochs):
            self.epoch += 1
            self.adjust()
            user_indices = streamer.next_train_user_batch()
            model(user_indices)
            if opts.save:
                self.val()

    def val(self):
        opts, model, streamer, epoch = self._options, self.model, self.streamer, self.epoch
        val_idx = 0
        scores_hat = []
        scores = []
        all_user_vec = model.test()
        while True:
            val_idx += 1
            data = streamer.next_test_batch()
            if data is None:
                break
            users, items, score = data
            score_hat = []
            for user, item in zip(users, items):
                score_hat.append(np.clip(all_user_vec[user, item], 0, 4))
            score_hat = np.array(score_hat)
            assert score_hat.ndim == 1
            scores_hat.extend(list(score_hat))
            scores.extend(list(score))
        mae = np.mean(np.abs(scores_hat - scores))
        print('[Test][MAE:{:.5f}]'.format(mae))
        scores_hat = np.array(scores_hat).astype(np.float32)
        scores = np.array(scores).astype(np.float32)
        val_score = np.sqrt(np.mean(np.power(scores_hat-scores, 2)))
        print('[Val][MAE:{:.5f}][MSE:{:.5f}]'.format(mae, val_score))
        if val_score < self.best_score:
            self.best_score = val_score
            path = os.path.join(self.save_folder, self.model_name + '_BestModel.pkl')
            print('[Val][Generate][NewBestScore: {}][SavePath: {}]'.format(val_score, path))
            torch.save(self.model.state_dict(), path)

    def test(self, from_file=True):
        opts, model, streamer = self._options, self.model, self.streamer
        if from_file:
            if opts.pre_path is None:
                path = os.path.join(self.save_folder, self.model_name + '_BestModel.pkl')
            else:
                path = opts.pre_path
            print('[Test][LoadPath: {}]'.format(path))
            model.load_state_dict(torch.load(path))
        val_idx = 0
        scores_hat = []
        scores = []
        self.adjust()
        all_user_vec = model.test()
        while True:
            val_idx += 1
            data = streamer.next_test_batch()
            if data is None:
                break
            users, items, score = data
            score_hat = []
            for user, item in zip(users, items):
                score_hat.append(np.clip(all_user_vec[user, item], 0, 4))
            score_hat = np.array(score_hat)
            assert score_hat.ndim == 1
            scores_hat.extend(list(score_hat))
            scores.extend(list(score))
        scores_hat = np.array(scores_hat).astype(np.float32)
        scores = np.array(scores).astype(np.float32)
        mse = np.sqrt(np.mean(np.power(scores_hat-scores, 2)))
        print('[Test][RMSE:{:.5f}]'.format(mse))
        mae = np.mean(np.abs(scores_hat-scores))
        print('[Test][MAE:{:.5f}]'.format(mae))

    def adjust(self):
        epoch, opts = self.epoch, self._options
        opts.learning_rate = 5e-6
        opts.l1_factor = 1e-8
        opts.l2_factor = 1e-8
        opts.dropout_rate = 0.0
        opts.display_every = 50
        opts.batch_size = 64
        opts.save_every = 50
        opts.display = (epoch % opts.display_every) == 0
        # opts.visual = (epoch % opts.visual_every) == 0
        opts.save = (epoch % opts.save_every) == 0
        param_groups = self.model.optimizer.param_groups
        for param_group in param_groups:
            param_group['lr'] = opts.learning_rate
        if opts.display:
            print('[Adjust][Epoch:{}][LearningRate:{:.6f}][L1:{}][L2:{}][Dropout:{}]'.
                  format(epoch, opts.learning_rate, opts.l1_factor, opts.l2_factor, opts.dropout_rate))


def read_commands():
    data_root = os.path.abspath('../../Data')
    parser = argparse.ArgumentParser(usage='Douban AutoEncoder Parameters')
    parser.add_argument('--is_training', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='AutoRecV0')
    parser.add_argument('--tag', type=str, default='V12')
    parser.add_argument('--device', type=str, default='5')
    parser.add_argument('--train_dat_path', type=str,
                        default=os.path.join(data_root, 'train/douban_train.txt'))
    # parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--test_dat_path', type=str,
                        default=os.path.join(data_root, 'train/douban_test.txt'))
    parser.add_argument('--user_emb_path', type=str,
                        default=os.path.join(data_root, 'train/douban_emb'))
    parser.add_argument('--log_dir', type=str, default=os.path.join(data_root, 'log'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(data_root, 'model'))
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--util_dir', type=str, default=os.path.join(data_root, 'util'))
    parser.add_argument('--util_folder', type=str, default=None)
    model_root = os.path.join(data_root, 'model')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--grad_value_clip', action='store_true', default=True)
    parser.add_argument('--grad_clip_value', type=float, default=0.2)
    parser.add_argument('--grad_norm_clip', action='store_true', default=True)
    parser.add_argument('--grad_norm_clip_value', type=float, default=2.0)
    parser.add_argument('--grad_global_norm_clip', action='store_true', default=False)
    parser.add_argument('--grad_global_norm_clip_value', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--pre_path', type=str, default=None)
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    main()
