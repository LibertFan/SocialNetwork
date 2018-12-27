import os
import numpy as np
import random
import argparse


class DataLoader(object):
    def __init__(self, opts):
        self._options = opts
        self.train_dat = self.load_dat(opts.train_dat_path)
        # 将训练数据转化成 用户*物品 的矩阵，对应位置上是用户对物品的评分
        self.stat = self.stat_score(self.train_dat)
        opts.max_score = np.max(self.stat)
        print('[MaxScore:{}]'.format(opts.max_score))
        # 确定用户数量和物品数量 # train_user_size train_item_size
        self.user_num, self.item_num = self.stat.shape
        print('[UserNum:{}][ItemNum:{}]'.format(self.user_num, self.item_num))
        # 加入用户序列号
        self.train_user_indices = np.arange(self.user_num).astype(np.int64)
        # 从训练数据当中获取数据并将这些转化为用户-物品数据和物品-用户数据两类
        self.train_user_dat = self.stat.copy()
        self.train_item_dat = np.transpose(self.stat).copy()
        # 确定每个用户打分的均值
        opts.mean_user_score = np.sum(np.clip(self.train_user_dat, 0, np.inf), 1) / np.clip(
            np.sum((self.train_user_dat > -0.5).astype(np.float32), 1), 1e-10, np.inf)
        opts.mean_item_score = np.sum(np.clip(self.train_item_dat, 0, np.inf), 1) / np.clip(
            np.sum((self.train_item_dat > -0.5).astype(np.float32), 1), 1e-10, np.inf)
        # 确定不同分数的个数
        self.score_size = np.max(self.stat) - np.min(self.stat) + 1
        # 在option当中加入这些变量
        opts.user_size = self.user_num
        opts.item_size = self.item_num
        opts.score_size = self.score_size
        # 加载用户社交网络的嵌入向量，对于没有社交信息的用户，使用有社交信息的人的向量的均值来代替
        self.user2vec, self.train_user_mask = self.load_user_emb(opts.user_emb_path)
        # 在option当中加入前途向量的维度
        opts.user_emb_size = self.user2vec.shape[-1]
        # self.user_cluster = self.load_cluster(opts.social_info_path)
        # 加载测试集，不需要对测试集进行特殊处理
        self.test_dat = self.load_dat(opts.test_dat_path)
        # 确定测试集的大小
        self.test_size = len(self.test_dat)
        # 定义训练集的用户和物品的指针以及测试集上的指针
        self.train_user_iter = self.train_item_iter = self.test_iter = -1

    @staticmethod
    def load_dat(dat_path):
        dat = []
        with open(dat_path, 'r') as f:
            records = f.readlines()
            for record in records:
                user_id, item_id, score = map(int, record.split('\t'))
                # 用户的整数编码需要在原编码的基础上减一
                dat.append([user_id-1, item_id-1, (score-1)])
        f.close()
        return np.array(dat)

    def load_user_emb(self, user_emb_path):
        user_emb_dict = dict()
        with open(user_emb_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                elem = line.split(' ')
                # 用户的整数编码需要在原编码的基础上减一
                user_id = int(elem[0]) - 1
                emb = np.array([float(val) for val in elem[1:]])
                user_emb_dict[user_id] = emb
        mean_emb = np.mean(np.array(list(user_emb_dict.values())), 0)
        user_mask = np.zeros([self.user_num]).astype(np.int64)
        for index in list(user_emb_dict.keys()):
            user_mask[index] = 1
        user_emb = np.array([user_emb_dict.get(user_id, mean_emb) for user_id in range(self.user_num)])
        f.close()
        return user_emb, user_mask

    def load_cluster(self, social_info_path):
        cluster = np.zeros([self.user_num, self.user_num])
        with open(social_info_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                user_id, follower_id = [int(v)-1 for v in line.split('\t')]
                cluster[user_id, follower_id] = 1.0
        f.close()
        return cluster

    @staticmethod
    def stat_score(dat):
        user_num = 1000
        item_num = 1000
        count = np.zeros([user_num, item_num]) - 1
        for record in dat:
            user_id, item_id, score = record
            count[int(user_id), int(item_id)] = score
        # count = count.astype(np.int32)
        return count

    def next_train_batch(self):
        user_dat = self.next_train_user_batch()
        user_dat = list(user_dat)
        item_dat = self.next_train_item_batch()
        user_dat.append(item_dat)
        return user_dat

    def next_train_user_batch(self):
        opts = self._options
        if self.train_user_iter == -1:
            random.shuffle(self.train_user_indices)
        self.train_user_iter += 1
        start_idx = self.train_user_iter * opts.batch_size
        end_idx = (self.train_user_iter + 1) * opts.batch_size
        if start_idx >= self.user_num:
            self.train_user_iter = -1
            return self.next_train_user_batch()
        else:
            user_indices = self.train_user_indices[start_idx:end_idx]
            # user_masks = self.train_user_mask[user_indices]
            # user_items = self.train_user_dat[user_indices]
            return user_indices

    def next_train_item_batch(self):
        opts = self._options
        if self.train_item_iter == -1:
            random.shuffle(self.train_item_dat)
        self.train_item_iter += 1
        start_idx = self.train_item_iter * opts.batch_size
        end_idx = (self.train_item_iter + 1) * opts.batch_size
        if start_idx >= self.item_num:
            self.train_item_iter = -1
            return self.next_train_item_batch()
        else:
            item_users = self.train_item_dat[start_idx:end_idx]
            return item_users

    def next_test_batch(self):
        opts = self._options
        self.test_iter += 1
        start_idx = self.test_iter * opts.batch_size
        end_idx = (self.test_iter + 1) * opts.batch_size
        if start_idx >= self.test_size:
            self.test_iter = -1
            return None
        else:
            dat = self.test_dat[start_idx:end_idx]
            user, item, score = dat[:, 0].astype(np.int32), dat[:, 1].astype(np.int32), dat[:, 2]
            return user, item, score


def read_commands():
    data_root = os.path.abspath('../../Data')
    parser = argparse.ArgumentParser(usage='Douban AutoEncoder Parameters')
    parser.add_argument('--is_training', action='store_true', default=True)
    parser.add_argument('--model_name', type=str, default='ClusterAEV0')
    parser.add_argument('--tag', type=str, default='V1')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--train_dat_path', type=str,
                        default=os.path.join(data_root, 'train/douban_train.txt'))
    # parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--test_dat_path', type=str,
                        default=os.path.join(data_root, 'train/douban_test.txt'))
    parser.add_argument('--user_emb_path', type=str,
                        default=os.path.join(data_root, 'train/douban_emb'))
    parser.add_argument('--social_info_path', type=str,
                        default=os.path.join(data_root, 'train/social_info.txt'))
    parser.add_argument('--log_dir', type=str, default=os.path.join(data_root, 'log'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(data_root, 'model'))
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--util_dir', type=str, default=os.path.join(data_root, 'util'))
    parser.add_argument('--util_folder', type=str, default=None)
    model_root = os.path.join(data_root, 'model')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    opts = parser.parse_args()
    return opts


def main():
    opts = read_commands()
    streamer = DataLoader(opts)
    user_cluster = streamer.user_cluster
    print('UserCluster', user_cluster.shape)
    cluster_num = np.sum(user_cluster, 1)

    print(np.max(cluster_num))
    print(np.min(cluster_num))
    print(np.mean(cluster_num))
    for _ in range(5):
        train_dat = streamer.next_train_user_batch()
        print(train_dat[:10])
        print(streamer.train_user_dat[train_dat[:10]])
        # for val in train_dat:
        #     print(val.shape)
        #     print(val[0])
        print('-'*100)
    print('='*100)
    test_dat = streamer.next_test_batch()
    for val in test_dat:
        print(val.shape)
        print(val[0])


if __name__ == '__main__':
    main()
