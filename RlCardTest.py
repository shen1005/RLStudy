import gym
import random
import numpy as np

cardDict = {0: None, 1: "R2", 2: "R3", 3: "R4", 4: "R5", 5: "R6", 6: "R7", 7: "R8", 8: "R9", 9: "R10", 10: "RJ", 11: "RQ", 12: "RK",
            13: "B2", 14: "B3", 15: "B4", 16: "B5", 17: "B6", 18: "B7", 19: "B8", 20: "B9", 21: "B10", 22: "BJ", 23: "BQ", 24: "BK",
            25: "M2", 26: "M3", 27: "M4", 28: "M5", 29: "M6", 30: "M7", 31: "M8", 32: "M9", 33: "M10", 34: "MJ", 35: "MQ", 36: "MK",
            37: "F2", 38: "F3", 39: "F4", 40: "F5", 41: "F6", 42: "F7", 43: "F8", 44: "F9", 45: "F10", 46: "FJ", 47: "FQ", 48: "FK",}
class deZhouEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=52, shape=(7,), dtype=int)
        # 现在的筹码池
        self.chips_pool = 0
        self.money = 0
        self.num = 2
        self.cardDict = {0: None, 1: "R2", 2: "R3", 3: "R4", 4: "R5", 5: "R6", 6: "R7", 7: "R8", 8: "R9", 9: "R10", 10: "RJ", 11: "RQ", 12: "RK",
            13: "B2", 14: "B3", 15: "B4", 16: "B5", 17: "B6", 18: "B7", 19: "B8", 20: "B9", 21: "B10", 22: "BJ", 23: "BQ", 24: "BK",
            25: "M2", 26: "M3", 27: "M4", 28: "M5", 29: "M6", 30: "M7", 31: "M8", 32: "M9", 33: "M10", 34: "MJ", 35: "MQ", 36: "MK",
            37: "F2", 38: "F3", 39: "F4", 40: "F5", 41: "F6", 42: "F7", 43: "F8", 44: "F9", 45: "F10", 46: "FJ", 47: "FQ", 48: "FK",}

    # 随机初始化打乱一副牌, 返回一个长度为7的列表, 最后两张牌是庄家的
    def reset(self, **kwargs):
        self.cards = list(range(1, 53))
        random.shuffle(self.cards)
        # 两个人下注了
        self.chips_pool = 1.0
        self.money = 0.5
        self.num = 2
        # list转成np.ndarray
        return np.array(self.cards[:2 + self.num] + [0 for i in range(5 - self.num)])

    # 虚构一个随机的庄家
    def _step_action(self):
        # 0.1弃牌, 0.9下注
        # return 0 if random.random() < 0.1 else 1
        return 1

    def _compare(self, player, banker):
        # 比较大小
        # 0表示玩家赢，1表示庄家赢
        if sum(player) > sum(banker):
            return 0
        else:
            return 1

    # 看有几个同样数字
    def _numSame(self, cards):
        num = 0
        for i in range(6):
            for j in range(i + 1, 7):
                if cards[i] % 13 == cards[j] % 13:
                    num += 1
        return num

    def tongHuaShun(self, cards):
        return cards[0] / 13 == cards[1] / 13 and cards[1] / 13 == cards[2] / 13 and cards[2] / 13 == cards[3] / 13 and cards[3] / 13 == cards[4] / 13 and cards[4] / 13 == cards[5] / 13 and cards[5] / 13 == cards[6] / 13

    def getScore(self, cards):
        # 计算分数
        # 看谁的牌大
        for i in range(7):
            cards[i] = cards[i] % 13
        cards.sort()
        # 加一起
        score = sum(cards)
        return score, cards[0]

    def judge(self):
        # self.cards的前2张是玩家的牌, 最后两张是庄家的牌， 2张后面五张牌是公共牌
        # 判断胜负
        # 玩家的牌
        player = self.cards[:2 + self.num]
        # 庄家的牌
        banker = self.cards[2:7] + self.cards[-2:]
        # 比较大小
        # 0表示玩家赢，1表示庄家赢
        # 先检查同花顺
        if self.tongHuaShun(player) and not self.tongHuaShun(banker):
            return 0
        elif self.tongHuaShun(banker) and not self.tongHuaShun(player):
            return 1
        elif self.tongHuaShun(banker) and self.tongHuaShun(player):
            return banker[-1] % 13 > player[-1] % 13
        # 再检查相同数字
        numPlayer = self._numSame(player)
        numBanker = self._numSame(banker)
        if numPlayer > numBanker:
            return 0
        elif numPlayer < numBanker:
            return 1
        else:
            # 计算分数
            scorePlayer, minPlayer = self.getScore(player)
            scoreBanker, minBanker = self.getScore(banker)
            if scorePlayer > scoreBanker:
                return 0
            elif scorePlayer < scoreBanker:
                return 1
            else:
                if minPlayer > minBanker:
                    return 0
                else:
                    return 1

    # step, 0表示跟注，1表示加注， 2表示弃牌
    def step(self, action):
        reward = 0
        done = False
        info = {}
        if action == 0:
            self.money += 0.5
        if action == 1:
            done = True
        action_b = self._step_action()
        self.num += 1
        if done:
            reward = -self.money
        elif self.num == 6:
            # 判断胜负
            result = self.judge()
            if result == 0:
                reward = self.chips_pool
            else:
                reward = -self.money
            done = True
        elif action_b == 0:
            reward = self.chips_pool
            done = True
        else:
            reward = 0
            self.chips_pool += 0.5
        return np.array(self.cards[:2 + self.num] + [0 for i in range(5 - self.num)]), reward * 10, done, info, {}








