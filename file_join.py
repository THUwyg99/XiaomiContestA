import pandas as pd
import time

def write():
    start = time.time()
    u_col = ['uid', 'age', 'sex', 'history_click_list']
    u_dtype = {'uid': "int64", 'age': "int32", 'sex': "int32", 'history_click_list': "str"}
    g_col = ['gid', 'cid1', 'cid2', 'cid3', 'brand_id', 'merchant_id']
    # g_dtype = {'gid': "int64", 'cid1': "int32", 'cid2': "int32", 'cid3': "int32",
    #            'brand_id': "int32", 'merchant_id': "int32"}
    train_col = ['uid', 'gid', 'time', 'phone', 'version', 'ct', 'cv']
    # tr_dtype = {'uid': "int64", 'gid': "int64", 'time': "int32", 'phone': "int32",
    #             'version': "int32", 'ct': "int32", 'cv': "int32"}
    test_col = ['uid', 'gid', 'time', 'phone', 'version']
    # te_dtype = {'uid': "int64", 'gid': "int64", 'time': "int32", 'phone': "int32", 'version': "int32"}

    print("Loading user data, cost time %s"%(time.time() - start))
    tmp_u = pd.read_csv("user_data.csv",
                        delimiter=',',
                        engine="c",
                        header=None,
                        index_col=None,
                        names=u_col,
                        dtype=u_dtype,
                        usecols=['uid', 'age', 'sex'],
                        na_values=0
                        )

    print("Loading item data, cost time %s"%(time.time() - start))
    tmp_g = pd.read_csv("goods_data.csv",
                        delimiter=',',
                        engine="c",
                        header=None,
                        index_col=None,
                        names=g_col)

    print("Loading train data, cost time %s"%(time.time() - start))
    train = pd.read_csv("train_data.csv",
                        delimiter=',',
                        engine="c",
                        header=None,
                        index_col=None,
                        names=train_col)

    print("Loading test data, cost time %s"%(time.time() - start))
    test = pd.read_csv("test_data.csv",
                        delimiter=',',
                        engine="c",
                        header=None,
                        index_col=None,
                        names=test_col)
    train.fillna(method='ffill')
    test.fillna(method='ffill')
    # test['phone'].fillna(0)
    # test['version'].fillna(0)

    print("Join user to train..., cost time %s"%(time.time() - start))
    join_tr = train.join(tmp_u.set_index("uid"), on="uid", how="inner").join(tmp_g.set_index("gid"), on="gid", how="inner")
    # print("Join goods to train..., cost time %s"%(time.time() - start))
    # join_tr = join_tr.join(tmp_g.set_index("gid"), on="gid", how="inner")
    print("Writing to CSV, cost time %s"%(time.time() - start))
    join_tr.drop(["uid", "gid"], axis=1).to_csv("trainA.csv", sep=",", header=False, index=False)

    print("Join user to test..., cost time %s"%(time.time() - start))
    join_te = test.join(tmp_u.set_index("uid"), on="uid", how="inner").join(tmp_g.set_index("gid"), on="gid", how="inner")
    # print("Join goods to test..., cost time %s"%(time.time() - start))
    # join_te = join_te.join(tmp_g.set_index("gid"), on="gid", how="inner")
    print("Writing to CSV, cost time %s"%(time.time() - start))
    join_te.drop(["uid", "gid"], axis=1).to_csv("testA.csv", sep=",", header=False, index=False)
    #
    # print("Join user and item...")
    # join_tmp = tmp_u.join(tmp_g.set_index("pvid"),on="pvid",how="inner")
    # print("Writing to csv...")
    # # datapath: xxxx/nl_train.csv
    # drop_pvid_join = join_tmp.drop(["pvid"], axis=1)
    # join_tmp.to_csv("%s/%s_%s.csv"%(path,cn,mod), sep=",", header=False, index=False)
    # print("Done!")

if __name__ == '__main__':
    # path = sys.argv[1]
    # cn = sys.argv[2]
    # mod = sys.argv[3]
    write()
