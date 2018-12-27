import torch
import torch.nn.functional as func


class ClusterAE(torch.nn.Module):
    def __init__(self, opts, user_dat, user_cluster):
        super(ClusterAE, self).__init__()
        self._options = opts
        user_dat = torch.from_numpy(user_dat).long()
        user_cluster = torch.from_numpy(user_cluster).long()
        if torch.cuda.is_available():
            user_dat = user_dat.cuda()
            user_cluster = user_cluster.cuda()
        self.user_dat = user_dat
        self.user_cluster = user_cluster
        self.user_auto_encoder = UserAutoEncoder(opts)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=opts.learning_rate)

    def forward(self, sample_user_indices):
        # 输入的只是采样出来的用户的index
        opts = self._options
        self.train()
        user_indices = torch.from_numpy(sample_user_indices)
        if torch.cuda.is_available():
            user_indices = user_indices.cuda()
        # 选出来用户对物品的打分求出来
        user_score = self.user_dat.index_select(0, user_indices)
        # 把用户的关注的人选出来（社交信息）
        user_cluster = self.user_cluster.index_select(0, user_indices)
        user_vec, user_score_hat, user_cluster_hat = \
            self.user_auto_encoder(user_score, user_cluster)
        # 计算用户对物品打分的重建损失
        score_ae_loss = self.solve_ae_loss(user_score_hat, user_score)
        # 采样负样本，连同正样本来做成mask来计算
        cluster_mask = self.mask_sample(user_cluster)
        # 计算用户的社交信息的重建损失
        cluster_ae_loss = self.solve_ae_loss(user_cluster_hat, user_cluster, mask=cluster_mask)
        # 找出每一个用户的关注用户
        max_cluster_user_size = int(user_cluster.sum(-1).max().long().data.cpu().numpy())
        if max_cluster_user_size > 0:
            # 用每一个用户关注的用户的隐藏表示来作为正则项来意图使表示准确
            # 导数不向关注的用户的方向传导
            cluster_user_mask, cluster_user = user_cluster.topk(max_cluster_user_size, -1)
            cluster_user_score = self.user_dat.index_select(0, cluster_user.view(-1)).\
                view(-1, max_cluster_user_size, opts.item_size)
            cluster_user_cluster = self.user_cluster.index_select(0, cluster_user.view(-1)).\
                view(-1, max_cluster_user_size, opts.user_size)
            cluster_sim = cluster_user_score.float().mul(user_score.float().unsqueeze(1)).sum(-1).\
                div(cluster_user_score.float().norm(2, -1).mul(user_score.float().norm(2, -1).unsqueeze(1))).\
                mul(cluster_user_mask.float()).clamp(0.1)
            cluster_weight = cluster_sim.div(cluster_sim.sum(-1, keepdim=True)).mul(cluster_user_mask.float())
            cluster_user_vec = self.user_auto_encoder(cluster_user_score, cluster_user_cluster, no_grad=True)
            sim_dist = cluster_user_vec.detach().sub(user_vec.unsqueeze(1)).norm(2, -1)
            cluster_reg_loss = opts.cluster_reg_factor * sim_dist.mul(cluster_weight).sum(-1).mean()
            # user_loss += cluster_reg_loss
        else:
            cluster_reg_loss = 0.0
        reg_loss = self.solve_reg_loss(scope=self.user_auto_encoder)
        # 调整各个loss的权重！
        user_loss = opts.score_ae_factor * score_ae_loss + opts.cluster_ae_factor * cluster_ae_loss + \
            cluster_reg_loss + reg_loss
        # user_loss = score_ae_loss + reg_loss
        if opts.display:
            print('[User][ScoreAELoss:{:.5f}][ClusterAELoss:{:.5f}][ClusterRegLoss:{:.5f}]'
                  '[RegLoss:{:.5f}][UserLoss:{:.5f}]'.
                  format(score_ae_loss, cluster_ae_loss, cluster_reg_loss, reg_loss, user_loss))
        self.backward(user_loss)

    def test(self):
        opts = self._options
        self.eval()
        user_hat = self.user_auto_encoder(self.user_dat, self.user_cluster, no_grad=True, to_end=True)
        return user_hat.data.cpu().numpy()

    def mask_sample(self, user_cluster):
        opts = self._options
        user_cluster = user_cluster.detach()
        pos_mask = user_cluster.float().ge(0.5).float()
        every_pos_num = pos_mask.sum(-1).long()
        sample_num = min(int(every_pos_num.max().data.cpu().numpy()), 5)
        masks = torch.zeros_like(user_cluster).float().\
            scatter(1, user_cluster.le(0.5).float().multinomial(sample_num), 1.0).\
            mul(every_pos_num.float().ge(0.5).float().unsqueeze(-1)).\
            add(pos_mask)
        return masks

    @staticmethod
    def solve_sim_loss(score_hat, score):
        assert score_hat.size() == score.size()
        score = score.float()
        # 如果score当中的数据是完整的的话会出bug，最好提前设置好 min_val 防止bug
        min_val = (score.min() + 0.5).detach()
        mask = score.ge(min_val).float()
        sim_loss = score.detach().sub(score_hat).pow(2).mul(mask)
        sim_loss = sim_loss.sum(-1).mean()
        return sim_loss

    @staticmethod
    def solve_ae_loss(value_hat, value, is_mask=True, mask=None):
        assert value_hat.size() == value.size()
        value = value.float()
        ae_loss = value.detach().sub(value_hat).pow(2)
        if is_mask:
            if mask is None:
                # 如果score当中的数据是完整的的话会出bug，最好提前设置好 min_val 防止bug
                min_val = (value.min() + 0.5).detach()
                mask = value.ge(min_val).float()
            else:
                if mask.dim() < ae_loss.dim():
                    mask = mask.unsqueeze(-1)
            ae_loss = ae_loss.mul(mask.float())
        ae_loss = ae_loss.sum(-1).mean()
        return ae_loss

    def solve_reg_loss(self, scope=None):
        opts, l1_loss_sum, l2_loss_sum = self._options, 0.0, 0.0
        if scope is None:
            named_parameters = self.named_parameters()
        else:
            named_parameters = scope.named_parameters()
        for name, param in named_parameters:
            l1_loss_sum += param.abs().sum()
            l2_loss_sum += param.pow(2).sum()
        reg_loss = opts.l1_factor * l1_loss_sum + opts.l2_factor * l2_loss_sum
        return reg_loss

    def backward(self, loss):
        opts = self._options
        optimizer = self.optimizer
        parameter = self.parameters()
        optimizer.zero_grad()
        loss.backward()
        if opts.grad_value_clip:
            torch.nn.utils.clip_grad_value_(parameter, opts.grad_clip_value)
        elif opts.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(parameter, opts.grad_norm_clip_value)
        optimizer.step()


class UserAutoEncoder(torch.nn.Module):
    def __init__(self, opts):
        super(UserAutoEncoder, self).__init__()
        self._options = opts
        self.side_fc_l1 = torch.nn.Linear(opts.user_size, 1024)
        self.side_fc_l2 = torch.nn.Linear(1024, 256)

        self.fc_l1 = torch.nn.Linear(opts.item_size, 1024)
        self.fc_l2 = torch.nn.Linear(1024, 256)
        self.fc_l3 = torch.nn.Linear(256, 32)

        self.item_fc_l1 = torch.nn.Linear(32, 1024)
        self.item_fc_l2 = torch.nn.Linear(1024, 1024)
        self.item_fc_l3 = torch.nn.Linear(1024, opts.item_size)

        self.emb_fc_l1 = torch.nn.Linear(32, 1024)
        self.emb_fc_l2 = torch.nn.Linear(1024, 1024)
        self.emb_fc_l3 = torch.nn.Linear(1024, opts.user_size)

        self.dropout = torch.nn.Dropout(opts.dropout_rate)
        self.var_init()

    def forward(self, user, user_emb, no_grad=False, to_end=False):
        opts = self._options
        if not no_grad:
            self.train()
            user_vec = func.relu(self.fc_l2(self.dropout(
                func.relu(self.fc_l1(user.float()))
            )))
            emb_vec = func.relu(self.side_fc_l2(self.dropout(
                func.relu(self.side_fc_l1(user_emb.float()))
            )))
            user_vec = func.relu(self.fc_l3(self.dropout(user_vec.add(emb_vec))))
            user_hat = self.item_fc_l3(self.dropout(
                func.relu(self.item_fc_l2(self.dropout(
                    func.relu(self.item_fc_l1(self.dropout(user_vec)))
                )))
            ))
            emb_hat = self.emb_fc_l3(self.dropout(
                func.relu(self.emb_fc_l2(self.dropout(
                    func.relu(self.emb_fc_l1(self.dropout(user_vec)))
                )))
            ))
            return user_vec, user_hat, emb_hat
        else:
            self.eval()
            with torch.no_grad():
                user_vec = func.relu(self.fc_l2(self.dropout(
                    func.relu(self.fc_l1(user.float()))
                )))
                emb_vec = func.relu(self.side_fc_l2(self.dropout(
                    func.relu(self.side_fc_l1(user_emb.float()))
                )))
                user_vec = func.relu(self.fc_l3(self.dropout(user_vec.add(emb_vec))))
                if not to_end:
                    return user_vec
                else:
                    user_hat = self.item_fc_l3(self.dropout(
                        func.relu(self.item_fc_l2(self.dropout(
                            func.relu(self.item_fc_l1(self.dropout(user_vec)))
                        )))
                    ))
                    return user_hat.clamp(0, opts.max_score)

    def var_init(self):
        opts = self._options
        torch.nn.init.xavier_normal_(self.side_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.side_fc_l2.weight)
        torch.nn.init.xavier_normal_(self.fc_l1.weight)
        torch.nn.init.xavier_normal_(self.fc_l2.weight)
        torch.nn.init.xavier_normal_(self.fc_l3.weight)
        torch.nn.init.xavier_normal_(self.item_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.item_fc_l2.weight)
        torch.nn.init.xavier_normal_(self.item_fc_l3.weight)
        torch.nn.init.xavier_normal_(self.emb_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.emb_fc_l2.weight)
        torch.nn.init.xavier_normal_(self.emb_fc_l3.weight)
