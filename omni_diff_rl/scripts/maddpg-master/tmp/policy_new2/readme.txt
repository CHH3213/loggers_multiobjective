用的是_get_dist2()的obs
reward设置为
 if any([
            'trapped' in self.status,
            'catch it' in self.status
        ]):
            reward[0] = 1000
            reward[1] = 1000
            done = True
        else:
           reward[0] += -dist
           reward[1] += dist
           
最后效果：两个都在转圈圈
