import scipy.io as sio
file_path ="/home/caohuanhui/Downloads/capture&pursue/maddpg-master/save_data/训练数据/policy_22"
def save_data(file_name,**args):
    # file_name=file_path+'/reward.mat'
    for key,values in args.items():
        sio.savemat(file_name, {key:values})





if __name__=='__main__':
    # list1={'1':1,"2":5}
    # save_data(file_path,list1)
    data=sio.loadmat(file_path+'/network_vs_default-a4_2.4.mat')
    # data=sio.loadmat(file_path+'/rewards.mat')
    # print(data.shape)
    print(data)
    # list = [1,2,3,4,5,6]
    # l = [list[i] for i in range(len(list)) if i == 0 or i==1 or i == 3 or i==5]
    # print(l)
    # p_x = obs_n[0][0] + obs_n[0][2]
    # p_y = obs_n[0][1] + obs_n[0][3]
    # v_x = obs_n[0][4] + obs_n[0][6]
    # v_y = obs_n[0][5] + obs_n[0][7]
    # obs_n[0][2] = p_x
    # obs_n[0][3] = p_y
    # obs_n[0][6] = v_x
    # obs_n[0][7] = v_y
    # # --------------------
    # # 取绝对位置绝对速度时
    # position.append(obs_n[0][0:4])
    # volocity.append(obs_n[0][4:8])

