追击者默认策略，训练逃跑者，有效果
连续动作空间--1维
reward设置：
	    def _compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        # rospy.logdebug("\nStart Computing Reward")
        dist = self._get_dist2()
        reward, done = np.zeros(2), False
        if any([
            'trapped' in self.status,
            'catch it' in self.status
        ]):
            reward[0] = 1000
            reward[1] = 1000
            done = True
        # else:
        #     reward[0] += -dist#-abs(2*self.theta)
        #     reward[1] += dist#+abs(2*self.theta)
        # elif dist>0.55 and dist<3:
        #     reward[0] =-abs(2*self.theta)
        #     reward[1] =abs(2*self.theta)
        else:
            reward[0] -= dist
            reward[1] += dist
        # rospy.logdebug("\nEnd Computing Reward\n")

        return reward, done
抓住时，reward误设置成两个都是1000！！！

