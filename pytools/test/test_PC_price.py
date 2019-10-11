class Thing(object):

    def __init__(self):
        self.cpu = 0
        self.gpu = 0
        self.motherboard = 0
        self.box = 0
        self.cooler = 0
        self.SSD = 0
        self.HDD = 0
        self.memory = 0
        self.power = 0

    @property
    def cheapest(self):
        return self.motherboard + self.box + self.power + self.memory + self.HDD

    @property
    def cheapest_SSD(self):
        return self.motherboard + self.box + self.power + self.memory + self.SSD

    @property
    def gpu_SSD(self):
        return self.motherboard + self.box + self.power + self.memory + self.SSD + self.gpu


t = Thing()
t.motherboard = 1659
t.memory = 319
t.SSD = 779
t.box = 189
t.power = 279
t.gpu = 1599

print("price SSD: {}".format(t.cheapest_SSD))
print("price gpu + SSD: {}".format(t.gpu_SSD))
