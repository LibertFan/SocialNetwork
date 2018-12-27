import torch
import torch.nn.functional as func


class AutoRec(torch.nn.Module):
    def __init__(self, opts, user_dat):
        super(AutoRec, self).__init__()
        self._options = opts
        user_dat = torch.from_numpy(user_dat).long()
        if torch.cuda.is_available():
            user_dat = user_dat.cuda()
        self.user_dat = user_dat
        self.user_auto_encoder = UserAutoEncoder(opts)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=opts.learning_rate)

    def forward(self, user_indices):
        self.train()
        opts = self._options
        user_indices = torch.from_numpy(user_indices)
        if torch.cuda.is_available():
            user_indices = user_indices.cuda()
        user_item_vec = self.user_dat.index_select(0, user_indices.long())
        # 对用户做自编码
        user_vec, user_item_vec_hat = self.user_auto_encoder(user_item_vec)
        # 计算求出来的分数和真实分数的平方差距
        rank_ae_loss = self.solve_ae_loss(user_item_vec_hat, user_item_vec)
        reg_loss = self.solve_reg_loss(scope=self.user_auto_encoder)
        user_loss = rank_ae_loss + reg_loss
        if opts.display:
            print('[User][AEUserItemLoss:{:.5f}][RegLoss:{:.5f}][UserLoss:{:.5f}]'.
                  format(rank_ae_loss, reg_loss, user_loss))
        self.backward(user_loss)

    def test(self):
        opts = self._options
        self.eval()
        all_user_score = self.user_auto_encoder(self.user_dat, no_grad=True, to_end=True).clamp(0, opts.max_score)
        return all_user_score.data.cpu().numpy()

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
        self.fc_l1 = torch.nn.Linear(opts.item_size, 1024)
        self.fc_l2 = torch.nn.Linear(1024, 1024)
        self.fc_l3 = torch.nn.Linear(1024, 32)
        self.rev_fc_l1 = torch.nn.Linear(32, 1024)
        self.rev_fc_l2 = torch.nn.Linear(1024, 1024)
        self.rev_fc_l3 = torch.nn.Linear(1024, opts.item_size)
        self.dropout = torch.nn.Dropout(opts.dropout_rate)
        self.var_init()

    def forward(self, user, no_grad=False, to_end=False):
        opts = self._options
        if not no_grad:
            user_vec = func.relu(self.fc_l2(self.dropout(
                func.relu(self.fc_l1(user.float()))
            )))
            user_vec = func.relu(self.fc_l3(self.dropout(user_vec)))
            user_hat = self.rev_fc_l3(self.dropout(
                func.relu(self.rev_fc_l2(self.dropout(
                    func.relu(self.rev_fc_l1(self.dropout(user_vec)))
                )))
            ))
            return user_vec, user_hat
        else:
            with torch.no_grad():
                user_vec = func.relu(self.fc_l3(
                    func.relu(self.fc_l2(
                        func.relu(self.fc_l1(user.float()))
                    ))
                ))
                if not to_end:
                    return user_vec
                else:
                    user_hat = self.rev_fc_l3(self.dropout(
                        func.relu(self.rev_fc_l2(self.dropout(
                            func.relu(self.rev_fc_l1(self.dropout(user_vec)))
                        )))
                    ))
                    return user_hat.clamp(0, opts.max_score)

    def var_init(self):
        opts = self._options
        torch.nn.init.xavier_normal_(self.fc_l1.weight)
        torch.nn.init.xavier_normal_(self.fc_l2.weight)
        torch.nn.init.xavier_normal_(self.fc_l3.weight)
        torch.nn.init.xavier_normal_(self.rev_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.rev_fc_l2.weight)
        torch.nn.init.xavier_normal_(self.rev_fc_l3.weight)
        # torch.nn.init.constant_(self.rev_fc_l3.bias, opts.mean_item_score)
        # torch.nn.init.constant_(self.rev_fc_l3.bias, 2.0)
