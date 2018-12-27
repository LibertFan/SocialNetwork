import torch
import torch.nn.functional as func


class AutoEncoder(torch.nn.Module):
    def __init__(self, opts, user_dat, item_dat, user2vec=None):
        super(AutoEncoder, self).__init__()
        self._options = opts
        self.user2vec = user2vec
        self.user_emb = torch.nn.Embedding(opts.user_size, opts.user_emb_size)
        self.user_emb.from_pretrained(embeddings=torch.from_numpy(self.user2vec), freeze=True)
        user_dat = torch.from_numpy(user_dat).long()
        item_dat = torch.from_numpy(item_dat).long()
        if torch.cuda.is_available():
            user_dat = user_dat.cuda()
            item_dat = item_dat.cuda()
        self.user_dat = user_dat
        self.item_dat = item_dat
        self.user_auto_encoder = UserAutoEncoder(opts)
        self.user_optimizer = torch.optim.SGD(params=self.user_auto_encoder.parameters(),
                                              lr=opts.learning_rate, momentum=0.9)

    def forward(self, sample_user_indices, sample_user_masks, sample_user, sample_item):
        self.train()
        user_loss = self.user_forward(sample_user_indices, sample_user_masks, sample_user)
        self.backward(user_loss, 'user')

    def test(self):
        opts = self._options
        user_indices = torch.arange(opts.user_size).long()
        all_user_emb = self.user_emb(user_indices)
        self.eval()
        all_user_vec, user_emb_hat, user_hat = self.user_auto_encoder(all_user_emb, self.user_dat)
        return user_hat

    def user_forward(self, sample_user_indices, sample_user_mask, sample_user):
        opts = self._options
        sample_user = torch.from_numpy(sample_user).long()
        sample_user_indices = torch.from_numpy(sample_user_indices).long()
        sample_user_mask = torch.from_numpy(sample_user_mask).long()
        if torch.cuda.is_available():
            sample_user = sample_user.cuda()
            sample_user_indices = sample_user_indices.cuda()
            sample_user_mask = sample_user_mask.cuda()
        # 将用户作嵌入
        sample_user_emb = self.user_emb(sample_user_indices)
        # 对用户做自编码
        sample_user_vec, sample_user_emb_hat, sample_user_item_hat = self.user_auto_encoder(
            sample_user_emb, sample_user)

        # 求自编码器的 reconstruction loss
        rank_ae_loss = self.solve_ae_loss(sample_user_item_hat, sample_user)

        emb_ae_loss = self.solve_ae_loss(sample_user_emb_hat, sample_user_emb, mask=sample_user_mask)
        reg_loss = self.solve_reg_loss(scope=self.user_auto_encoder)
        user_loss = rank_ae_loss + emb_ae_loss + reg_loss
        if opts.display:
            print('[User][AEUserItemLoss:{:.5f}][AEUserEmbLoss:{:.5f}][RegLoss:{:.5f}]'
                  '[UserLoss:{:.5f}]'.
                  format(rank_ae_loss, emb_ae_loss, reg_loss, user_loss))
        return user_loss

    @staticmethod
    def solve_sim_loss(score_hat, score):
        assert score_hat.size() == score.size()
        score = score.float()
        mask = score.ge(-0.5).float()
        sim_loss = score.detach().sub(score_hat).mul(mask.detach()).pow(2)
        sim_loss = sim_loss.sum(-1).mean()
        return sim_loss

    @staticmethod
    def solve_ae_loss(value_hat, value, is_mask=True, mask=None):
        # 需要重新把自编码器的mask重新做好
        assert value_hat.size() == value.size()
        value = value.float()
        ae_loss = value.detach().sub(value_hat).pow(2)
        if is_mask:
            if mask is None:
                mask = value.ge(-0.5).float()
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

    def backward(self, loss, mode='user'):
        opts = self._options
        if mode == 'user':
            optimizer = self.user_optimizer
            parameter = self.user_auto_encoder.parameters()
        else:
            raise Exception('Current mode is not supported!')
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
        # self.user2vec = user2vec
        # self.user_emb = torch.nn.Embedding(opts.user_size, opts.user_emb_size)
        self.side_fc_l1 = torch.nn.Linear(opts.user_emb_size, 1024)
        self.side_fc_l2 = torch.nn.Linear(1024, 256)
        self.fc_l1 = torch.nn.Linear(opts.item_size, 1024)
        self.fc_l2 = torch.nn.Linear(1024, 256)
        self.fc_l3 = torch.nn.Linear(256, 32)
        self.rev_fc_l1 = torch.nn.Linear(32, 256)
        self.rev_fc_l2 = torch.nn.Linear(256, 1024)
        self.rev_fc_l3 = torch.nn.Linear(1024, opts.item_size+opts.user_emb_size)
        self.var_init()

    def forward(self, user_emb, user, no_grad=False):
        opts = self._options
        if not no_grad:
            side_vec = func.relu(self.side_fc_l2(func.relu(self.side_fc_l1(user_emb))))
            user_vec = func.relu(self.fc_l2(func.relu(self.fc_l1(user.float()))))
            user_vec = func.relu(self.fc_l3(side_vec.add(user_vec)))
            user_hat = self.rev_fc_l3(func.relu(self.rev_fc_l2(func.relu(self.rev_fc_l1(user_vec)))))
            user_item_hat = user_hat[:, :opts.item_size]
            user_emb_hat = user_hat[:, opts.item_size:]
            return user_vec, user_emb_hat, user_item_hat
        else:
            with torch.no_grad():
                side_vec = func.relu(self.side_fc_l2(func.relu(self.side_fc_l1(user_emb))))
                user_vec = func.relu(self.fc_l2(func.relu(self.fc_l1(user.float()))))
                user_vec = func.relu(self.fc_l3(side_vec.add(user_vec)))
                return user_vec

    def var_init(self):
        torch.nn.init.xavier_normal_(self.side_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.side_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.fc_l1.weight)
        torch.nn.init.xavier_normal_(self.fc_l2.weight)
        torch.nn.init.xavier_normal_(self.fc_l3.weight)
        torch.nn.init.xavier_normal_(self.rev_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.rev_fc_l2.weight)
        torch.nn.init.xavier_normal_(self.rev_fc_l3.weight)
