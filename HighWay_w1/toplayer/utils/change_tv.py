from collections import deque

class ChangeTV:
    
    def __init__(self):
        self.tvmap = {
            'w2' : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            'w3' : [13, 14, 15, 16, 17, 18, 19, 20, 21, 22], 
            'w4' : [23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 
            'w5' : [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            'full':[i for i in range(3, 43)]
        }
        self.tv_deq = deque(maxlen = 40)
        self.tv_change = 2
        for i in range(3, 43):
            self.tv_change += 1
            self.tv_deq.append(self.tv_change)
        print('ChangeTV init done')
    
    def step(self):
        self.tv_change += 1
        if self.tv_change == 87:
            self.tv_change = 1
        self.tv_deq.append(self.tv_change)
        lis_tar = list(self.tv_deq)
        lis_cur = []
        for key in self.tvmap.keys():
            if key != 'full':
                lis_cur += self.tvmap[key]
        rm_ele = list(set(lis_cur) - set(lis_tar))[0]
        ap_ele = list(set(lis_tar) - set(lis_cur))[0]

        rm_key = 'w2'
        for key in self.tvmap.keys():
            for i in self.tvmap[key]:
                if key!= 'full' and i == rm_ele:
                    rm_key =key

        lis_change = self.tvmap[rm_key]
        lis_change.remove(rm_ele)
        lis_change.append(ap_ele)
        self.tvmap[rm_key] = lis_change


        #print('change\t', rm_key, '\t', str(rm_ele), '-->', str(ap_ele))
        return rm_key, rm_ele, ap_ele

if __name__ == '__main__':
    change_tv = ChangeTV()
    for i in range(100):
        change_tv.step()
