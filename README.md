# SocialNetwork

Data Format:

douban_train.txt and douban_test.txt:

For each line:
userid /t itemid /t rating


social_info.txt:

For each line:
follower /t followee

(follower:关注者，followee:被关注者)

douban_emb:

first line: number of users, embedding dimensions
for each line:

userid dimension1 dimension2 ... dimension128




Evaluation:

Rooted Mean Square Error(RMSE) and Mean Avarage Error(MAE).
