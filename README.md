# SocialNetwork

## Data Format:

douban_train.txt and douban_test.txt:

For each line:
userid /t itemid /t rating


social_info.txt:

For each line:
follower /t followee

(follower:关注者，followee:被关注者)

douban_emb(叶蓉你不用管这个数据):

first line: number of users, embedding dimensions

for each line:

userid dimension1 dimension2 ... dimension128

\\
\\

## Evaluation:

Rooted Mean Square Error(RMSE) and Mean Avarage Error(MAE).

## 社交网络的可视化：
见social_visual中，graph-npm.html文件，效果如下图，记得将三个.js文件放到同一文件夹下。
![](/social_visual/visualization.png)